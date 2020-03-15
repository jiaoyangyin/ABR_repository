import os
import logging
import numpy as np
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import tensorflow as tf
import env
import a3c3 as a3c
import load_trace
import ABR
import rl_test

S_INFO = 4  #the number of state
S_LEN = 8 #the number of past states
A_DIM = 4
ACTOR_LR_RATE = 0.00001#0.0001  # parameters can be improved
CRITIC_LR_RATE = 0.0001#0.001  #parameters can be improved
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]
RANDOM_SEED = 42
M_IN_K = 1000.0
RAND_RANGE = 1000
LOG_FILE = './results/log'
SUMMARY_DIR = './results'
NN_MODEL = None
NETWORK_TRACE = 'mix'
VIDEO_TRACE = '/sports'
DEBUG = True
#LOG_FILE_PATH = './log/'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'

GOP = 50
BW_NORM_FACTOR = 1500.0
BUFFER_NORM_FACTOR = 1.0

tf.estimator


def central_agent(net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=config) as sess, open(LOG_FILE + '_test', 'wb') as test_log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = 0

        # assemble experiences from agents, compute the gradients
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in xrange(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in xrange(NUM_AGENTS):
                # get state action and reward from agents
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            # compute aggregated gradient
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            # assembled_actor_gradient = actor_gradient_batch[0]
            # assembled_critic_gradient = critic_gradient_batch[0]
            # for i in xrange(len(actor_gradient_batch) - 1):
            #     for j in xrange(len(assembled_actor_gradient)):
            #             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
            #             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
            # actor.apply_gradients(assembled_actor_gradient)
            # critic.apply_gradients(assembled_critic_gradient)
            for i in xrange(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            epoch += 1
            avg_reward = total_reward / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)


def agent(agent_id, all_cooked_time, all_cooked_bw, all_file_names, net_params_queue, exp_queue):

    # create result directory
    #if not os.path.exists(LOG_FILE_PATH):
        #os.makedirs(LOG_FILE_PATH)

    # -- End Configuration --
    # You shouldn't need to change the rest of the code here.
    log_file_path = LOG_FILE + '_agent_' + str(agent_id)

    video_trace_prefix = './dataset/video_trace' + VIDEO_TRACE + '/new_frame_trace_'

    # load the trace
    #all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)
        # random_seed
    random_seed = agent_id
    count = 0
    trace_count = 1
    FPS = 25
    frame_time_len = 0.04
    reward_all_sum = 0
    run_time = 0
    # init
    # setting one:
    #     1,all_cooked_time : timestamp
    #     2,all_cooked_bw   : throughput
    #     3,all_cooked_rtt  : rtt
    #     4,agent_id        : random_seed
    #     5,logfile_path    : logfile_path
    #     6,VIDEO_SIZE_FILE : Video Size File Path
    #     7,Debug Setting   : Debug
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                                        all_cooked_bw=all_cooked_bw,
                                        random_seed=random_seed,
                                        logfile_path=log_file_path,
                                        VIDEO_SIZE_FILE=video_trace_prefix,
                                        Debug=DEBUG)

    BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]  # kpbs
    TARGET_BUFFER = [0, 0.04]  # seconds
    # ABR setting
    RESEVOIR = 0.5
    CUSHION = 2

    cnt = 0
    # defalut setting
    last_bit_rate = 0
    bit_rate = 0
    target_buffer = 1
    latency_limit = 4

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=config) as sess: #open(LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        actor_net_params, critic_net_params = net_params_queue.get()  # update the network  parameters from central agent
        actor.set_network_params(
            actor_net_params)  # this part is only initial the network parameters, they will be updated in the loop
        critic.set_network_params(critic_net_params)

        #last_bit_rate = DEFAULT_QUALITY
        #bit_rate = DEFAULT_QUALITY
        reward = 0.0
        action_vec = np.zeros(A_DIM)
        #action_vec[bit_rate] = 1
        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []
        # QOE setting
        reward_frame = 0
        reward_all = 0
        SMOOTH_PENALTY = 0
        REBUF_PENALTY = 7
        LANTENCY_PENALTY = 0.005
        SKIP_PENALTY = 0.5
        # past_info setting
        past_frame_num = 7500
        S_time = [0] * past_frame_num
        S_time_interval = [0] * past_frame_num
        S_send_data_size = [0] * past_frame_num
        S_chunk_len = [0] * past_frame_num
        S_rebuf = [0] * past_frame_num
        S_buffer_size = [0] * past_frame_num
        S_end_delay = [0] * past_frame_num
        S_chunk_size = [0] * past_frame_num
        S_play_time_len = [0] * past_frame_num
        S_decision_flag = [0] * past_frame_num
        S_buffer_flag = [0] * past_frame_num
        S_cdn_flag = [0] * past_frame_num
        S_skip_time = [0] * past_frame_num
        # params setting
        call_time_sum = 0
        time_previous = 0

        while True:
            reward_frame = 0

            time, time_interval, send_data_size, chunk_len, \
            rebuf, buffer_size, play_time_len, end_delay, \
            cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag, \
            buffer_flag, cdn_flag, skip_flag, end_of_video = net_env.get_video_frame(bit_rate, target_buffer,
                                                                                     latency_limit)

            S_time.pop(0)
            S_time_interval.pop(0)
            S_send_data_size.pop(0)
            S_chunk_len.pop(0)
            S_buffer_size.pop(0)
            S_rebuf.pop(0)
            S_end_delay.pop(0)
            S_play_time_len.pop(0)
            S_decision_flag.pop(0)
            S_buffer_flag.pop(0)
            S_cdn_flag.pop(0)
            S_skip_time.pop(0)

            S_time.append(time)
            S_time_interval.append(time_interval)
            S_send_data_size.append(send_data_size)
            S_chunk_len.append(chunk_len)
            S_buffer_size.append(buffer_size)
            S_rebuf.append(rebuf)
            S_end_delay.append(end_delay)
            S_play_time_len.append(play_time_len)
            S_decision_flag.append(decision_flag)
            S_buffer_flag.append(buffer_flag)
            S_cdn_flag.append(cdn_flag)
            S_skip_time.append(skip_frame_time_len)

            if end_delay <= 1.0:
                LANTENCY_PENALTY = 0.005
            else:
                LANTENCY_PENALTY = 0.01

            if not cdn_flag:
                reward_frame = 0.7 * frame_time_len * float(BIT_RATE[
                                                          bit_rate]) / 1000 - REBUF_PENALTY * rebuf
                reward += reward_frame
            else:
                reward_frame = -(REBUF_PENALTY * rebuf)
                reward += reward_frame
            if decision_flag or end_of_video:
                # reward formate = play_time * BIT_RATE - 4.3 * rebuf - 1.2 * end_delay
                reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)

                if abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000 >= 1.0:
                    SMOOTH_PENALTY += 0

                elif abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000 < 1.0 and SMOOTH_PENALTY > 0.1:
                    SMOOTH_PENALTY -= 0

                if np.sum(S_rebuf[-51:-1]) >= 1 and REBUF_PENALTY <= 10.0:
                    REBUF_PENALTY += 0.3

                elif np.sum(S_rebuf[-51:-1]) < 1 and REBUF_PENALTY > 0.1:
                    REBUF_PENALTY -= 0.1

                reward += reward_frame
                # last_bit_rate
                last_bit_rate = bit_rate

                r_batch.append(reward)

                reward = 0.0

                # retrieve previous state
                if len(s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(s_batch[-1], copy=True)

                # dequeue history record
                state = np.roll(state, -1, axis=1)
                #A = S_buffer_size[-51:-1]
                # this should be S_INFO number of terms
                #T_all = float(np.sum(S_time_interval[-51:-1]))
                T_all = time - time_previous
                time_previous = time
                num_of_frame = float(GOP / T_all)
                #print 'number of frames:', num_of_frame
                throughput = float(np.sum(S_send_data_size[-51:-1]))/float(np.sum(S_time_interval[-51:-1]))
                state[0, -1] = BIT_RATE[bit_rate] / float(np.max(BIT_RATE))  # last quality present
                state[1, -1] = num_of_frame / FPS
                state[2, -1] = throughput / M_IN_K / BW_NORM_FACTOR# kilo byte / ms #history
                #state[3, -1] = np.sum(S_skip_time[-51:-1]) / BUFFER_NORM_FACTOR #skip frame #present
                #state[4, -1] = S_end_delay[-1] / BUFFER_NORM_FACTOR  # latency #present
                state[3, -1] = np.sum(S_rebuf[-51:-1]) / BUFFER_NORM_FACTOR

                #print 'state1:', (num_of_frame / FPS)
                #state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

                # compute action probability vector
                action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                bit_rate = np.argmax(action_prob)
                print("bitrate: ", BIT_RATE[bit_rate])
                #action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                #action_cumsum = np.cumsum(action_prob)
                #bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                #print 'bitrate: ', bit_rate
                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states

                entropy_record.append(a3c.compute_entropy(action_prob[0]))

                if (len(r_batch) >= TRAIN_SEQ_LEN) or end_of_video:  # if the number of trained chunks is up to 100 or it is at the end of the video, the state, action, reward and entropy will be sent to central agent
                    if len(r_batch) > 1:
                        exp_queue.put([s_batch[1:],  # ignore the first chuck
                                       a_batch[1:],  # since we don't have the
                                       r_batch[1:],  # control over it
                                      end_of_video,
                                      {'entropy': entropy_record}])

                        # synchronize the network parameters from the coordinator
                        actor_net_params, critic_net_params = net_params_queue.get()
                        actor.set_network_params(actor_net_params)
                        critic.set_network_params(critic_net_params)

                        del s_batch[:]
                        del a_batch[:]
                        del r_batch[:]
                        del entropy_record[:]

            if end_of_video:
                print("network traceID, network_reward, avg_running_time", trace_count, reward_all)#, call_time_sum / cnt)
                reward_all_sum += reward_all
                #run_time += call_time_sum / cnt
                #if trace_count >= len(all_file_names):
                 #   trace_count = 1
                    #break
                trace_count += 1
                cnt = 0

                call_time_sum = 0
                last_bit_rate = 0
                reward_all = 0
                bit_rate = 0
                target_buffer = 0

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)

                S_time_interval = [0] * past_frame_num
                S_send_data_size = [0] * past_frame_num
                S_chunk_len = [0] * past_frame_num
                S_rebuf = [0] * past_frame_num
                S_buffer_size = [0] * past_frame_num
                S_end_delay = [0] * past_frame_num
                S_chunk_size = [0] * past_frame_num
                S_play_time_len = [0] * past_frame_num
                S_decision_flag = [0] * past_frame_num
                S_buffer_flag = [0] * past_frame_num
                S_cdn_flag = [0] * past_frame_num
            else:
                if decision_flag:
                    s_batch.append(state)

                    action_vec = np.zeros(A_DIM)
                    action_vec[bit_rate] = 1
                    a_batch.append(action_vec)

            reward_all += reward_frame

def main():

    np.random.seed(RANDOM_SEED) #generate a random number
    assert len(BIT_RATE) == A_DIM  #if true get 1 else get AssertionError

    # create result directory
    if not os.path.exists(SUMMARY_DIR):  #create result dictionary
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in xrange(NUM_AGENTS):  #the main function is the main process and create queues in parent process
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    network_trace_dir = './dataset/network_trace/' + NETWORK_TRACE + '/'
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)
    #all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    for i in xrange(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,all_file_names,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in xrange(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__': #has the same function as C main()
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    for d in ['/device:GPU:5']:
        with tf.device(d):
            main()                   #just a function's name

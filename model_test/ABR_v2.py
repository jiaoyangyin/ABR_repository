import tensorflow as tf
import a3c3 as a3c
import a3c3_1 as a3c1
import a3c3_2 as a3c2
import a3c3_3 as a3c3
import numpy as np
import model
import csv


TARGET_BUFFER = [0, 0.04]
S_INFO = 4  #the number of state
S_LEN = 8 #the number of past states
A_DIM = 4
BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]
RAND_RANGE = 1000
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001

"""
NN_MODEL = "./fixed/nn_model_ep_10800.ckpt" # model path settings
NN_MODEL1 = "./high/nn_model_ep_40400.ckpt"
NN_MODEL2 = "./low/nn_model_ep_5100.ckpt"
NN_MODEL3 = "./medium/nn_model_ep_13400.ckpt"
"""


NN_MODEL = "./fixed/nn_model_ep_10800.ckpt" # model path settings
NN_MODEL1 = "./high/nn_model_ep_40400.ckpt"
NN_MODEL2 = "./low/nn_model_ep_5100.ckpt"
NN_MODEL3 = "./medium/nn_model_ep_13400.ckpt"


GOP = 50
BW_NORM_FACTOR = 1500.0
BUFFER_NORM_FACTOR = 1.0
FPS = 25
M_IN_K = 1000.0

throughput_list_avg = []
frame_count = 0
select_flag = 0
record_flag = 0
frames_size = 0
received_time = 0
previous_time = 0
saver = tf.train.Saver()
graph = tf.get_default_graph()

class Algorithm:
     def __init__(self):
     # fill your self params
         self.buffer_size = 0
         self.bit_rate = 0
         self.last_bit_rate = 0

         self.action_vec = np.zeros(A_DIM)
         self.action_vec[self.bit_rate] = 1
         self.s_batch = [np.zeros((S_INFO, S_LEN))]
         self.a_batch = [self.action_vec]
         self.r_batch = []
         self.entropy_record = []


     # Intial
     def Initial(self):
     # Initail your session or something
         #with tf.Session() as sess:
         self.sess = tf.Session()
         self.actor = a3c.ActorNetwork(self.sess,
                                       state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                       learning_rate=ACTOR_LR_RATE)

         self.sess.run(tf.global_variables_initializer())
         #saver = tf.train.Saver()  # save neural net parameters

     # restore neural net parameters
         nn_model = NN_MODEL
         if nn_model is not None:  # nn_model is the path to file
             saver.restore(self.sess, nn_model)
             print("Model restored.")
             #self.critic = a3c.CriticNetwork(sess,
                                         #state_dim=[S_INFO, S_LEN],
                                         #learning_rate=CRITIC_LR_RATE)

     #Define your al
     def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,S_end_delay, S_decision_flag, S_buffer_flag,S_cdn_flag,S_skip_time, end_of_video, cdn_newest_id,download_id,cdn_has_frame,IntialVars):

         # If you choose the marchine learning
         if len(self.s_batch) == 0:
             state = [np.zeros((S_INFO, S_LEN))]
         else:
             state = np.array(self.s_batch[-1], copy=True)

             # dequeue history record
         state = np.roll(state, -1, axis=1)
            # this should be S_INFO number of terms
         T_all = float(np.sum(S_time_interval[-51:-1]))
         if T_all > 0:
             num_of_frame = float(GOP / T_all)
         else:
             num_of_frame = 0
         # print 'number of frames:', num_of_frame
         if np.sum(S_time_interval[-51:-1]) > 0:
             throughput = float(np.sum(S_send_data_size[-51:-1])) / float(np.sum(S_time_interval[-51:-1]))
         else:
             throughput = 0
         state[0, -1] = BIT_RATE[self.bit_rate] / float(np.max(BIT_RATE))  # last quality present
         state[1, -1] = num_of_frame / FPS
         state[2, -1] = throughput / M_IN_K / BW_NORM_FACTOR  # kilo byte / ms #history
         state[3, -1] = np.sum(S_rebuf[-51:-1]) / BUFFER_NORM_FACTOR

         #network_condition = 0
         global frame_count, record_flag, select_flag, frames_size, previous_time
         if abs(time - int(round(time / 0.04)) * 0.04) > 1e-10:
             frame_count += 1

         #select_flag = time / 20
         if time > 2940.5 or time == 0:
             record_flag = 0
             select_flag = 0
             previous_time = 0

         print("time:", time)

         if int(time / 0.5) > record_flag:
             record_flag = time / 0.5
             print("record_flag:", record_flag)
             frames_size = float(np.sum(S_send_data_size[-frame_count:]))
             print("frames_size:", frames_size)
             received_time = float(np.sum(S_time_interval[-frame_count:]))
             print("received_time:", received_time)
             throughput_tmp = float(frames_size) / float(received_time) / 1000000
             throughput_list_avg.append(throughput_tmp)
             previous_time = time
             frame_count = 0

         """
         while 0 in S_send_data_size:
             print("S_send_data_size", S_send_data_size)
             S_send_data_size.remove(0)
         print("S_send_data_size", S_send_data_size)
         while 0 in S_time_interval:
             print("S_time_interval", S_time_interval)
             S_time_interval.remove(0)
         print("S_time_interval", S_time_interval)
         """

         #if len(S_send_data_size) >= 602 and len(S_send_data_size[102:]) % 500 == 0:
         #if frame_count % 500 == 0 and S_time_interval[-501] > 0
         if int(time / 20) > select_flag:
             select_flag = time / 20
             #frame_count = 0
             throughput_list = np.true_divide((np.array(S_send_data_size[-501:-1])).astype(float),
                                              (np.array(S_time_interval[-501:-1])).astype(float))
             throughput_list = [x / y for x in throughput_list for y in [1000000]]
             #print("S_send_data_size:", S_send_data_size[-501:-1])
             #print("S_time_interval:", S_time_interval[-501:-1])
             #print("throughput_list:", throughput_list)
             for i in range(40):
                 if i == 0:
                     start = i * 12 + 1
                     end = start + 12
                 elif i > 0 and i % 2 != 0:
                     start = end + 0
                     end = start + 12
                 elif i > 0 and i % 2 == 0:
                     start = end + 1
                     end = start + 12

                 #throughput_list_avg.append(float(np.sum(throughput_list[start:end])) / 12.0)
             #print("throughput_list_avg:", throughput_list_avg[-40:])
             network_condition = model.train(throughput_list_avg[-40:])
             with open("./train_data.csv", 'a') as t:
                 writer = csv.writer(t)
                 writer.writerow(throughput_list_avg[-40:])
             print("network_condition", network_condition)
             network_condition = np.argmax(network_condition)

             if network_condition == 0:
                 nn_model = NN_MODEL
                 #print("Model0 restored.")
             #self.actor = a3c.ActorNetwork(self.sess,
              #                             state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
               #                            learning_rate=ACTOR_LR_RATE)
             elif network_condition == 1:
                 nn_model = NN_MODEL1
                 #print("Model1 restored.")
             #self.actor = a3c.ActorNetwork(self.sess,
              #                             state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
               #                            learning_rate=ACTOR_LR_RATE)
             elif network_condition == 2:
                 nn_model = NN_MODEL2
                 #print("Model2 restored.")
             #self.actor = a3c.ActorNetwork(self.sess,
              #                             state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
               #                            learning_rate=ACTOR_LR_RATE)
             elif network_condition == 3:
                 nn_model = NN_MODEL3
                 #print("Model3 restored.")
             #self.actor = a3c.ActorNetwork(self.sess,
              #                             state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
               #                            learning_rate=ACTOR_LR_RATE)

             if nn_model is not None:
                 saver.restore(self.sess, nn_model)
                 print("NN_MODEL %g restored" % network_condition)
                 with open("./Model_selection.txt", 'a') as t1:
                     t1.write("NN_MODEL %g restored\n" % network_condition)
                     t1.close()

         if not S_decision_flag[-1]:
             return 0, 0, 0

         if S_decision_flag[-1]:
             # compute action probability vector
             with graph.as_default():
                 action_prob = self.actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
             #action_cumsum = np.cumsum(action_prob)
             self.bit_rate = np.argmax(action_prob)

             self.s_batch.append(state)

             self.entropy_record.append(a3c.compute_entropy(action_prob[0]))

             if end_of_video:
                 self.last_bit_rate = 0
                 self.bit_rate = 0  # use the default action here

                 del self.s_batch[:]
                 del self.a_batch[:]
                 del self.r_batch[:]

                 self.action_vec = np.zeros(A_DIM)
                 self.action_vec[self.bit_rate] = 1

                 self.s_batch.append(np.zeros((S_INFO, S_LEN)))
                 self.a_batch.append(self.action_vec)
                 self.entropy_record = []

             target_buffer = 1
             latency_limit = 4

             return self.bit_rate, target_buffer, latency_limit

         # If you choose other
         #......



     def get_params(self):
     # get your params
        your_params = []
        return your_params

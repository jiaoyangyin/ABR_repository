import tensorflow as tf
import a3c3 as a3c
import numpy as np


TARGET_BUFFER = [0, 0.033]
S_INFO = 4  #the number of state
S_LEN = 8 #the number of past states
A_DIM = 4
BIT_RATE = [2000.0, 2500.0, 3000.0, 3500.0]
RAND_RANGE = 1000
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NN_MODEL = "./nn_model_ep_20000.ckpt" # model path settings
GOP = 30
BW_NORM_FACTOR = 1500.0
BUFFER_NORM_FACTOR = 1.0
FPS = 30
M_IN_K = 1000.0

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
         saver = tf.train.Saver()  # save neural net parameters

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
         T_all = float(np.sum(S_time_interval[-61:-1]))
         num_of_frame = float(GOP / T_all)
         # print 'number of frames:', num_of_frame
         throughput = float(np.sum(S_send_data_size[-61:-1])) / float(np.sum(S_time_interval[-61:-1]))
         state[0, -1] = BIT_RATE[self.bit_rate] / float(np.max(BIT_RATE))  # last quality present
         state[1, -1] = num_of_frame / FPS
         state[2, -1] = throughput / M_IN_K / BW_NORM_FACTOR  # kilo byte / ms #history
         state[3, -1] = np.sum(S_rebuf[-61:-1]) / BUFFER_NORM_FACTOR

            # compute action probability vector
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

# import tensorflow as tf

# NN_MODEL = "./submit/results/nn_model_ep_18200.ckpt" # model path settings
TARGET_BUFFER = [0, 0.033]
BIT_RATE = [2000.0, 2500.0, 3000.0, 3500.0]
M_IN_K = 1000.0


class Algorithm:
    def __init__(self):
        # fill your self params
        self.buffer_size = 0

    # Intial
    def Initial(self):
        # Initail your session or something

        # restore neural net parameters
        self.buffer_size = 0

    # Define your al
    def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,
            S_end_delay, S_decision_flag, S_buffer_flag, S_cdn_flag, S_skip_time, end_of_video, cdn_newest_id,
            download_id, cdn_has_frame, IntialVars):

        # If you choose the marchine learning
        '''state = []

        state[0] = ...
        state[1] = ...
        state[2] = ...
        state[3] = ...
        state[4] = ...

        decision = actor.predict(state).argmax()
        bit_rate, target_buffer = decison//4, decison % 4 .....
        return bit_rate, target_buffer'''

        # If you choose BBA
        bit_rate = 3

        target_buffer = 1
        latency_limit = 4

        return bit_rate, target_buffer, latency_limit

        # If you choose other
        # ......

    def get_params(self):
        # get your params
        your_params = []
        return your_params

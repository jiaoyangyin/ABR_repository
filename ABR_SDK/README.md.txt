   * [Adaptive bitrate selection for live streaming based on reinforcement learning]
     * [综述]
	 * [依赖环境]
	 * [运行说明]
	 * [框架说明]
	 * [算法简介]
	 * [实验结果]
	 * [参考文献]
	 
# Adaptive bitrate selection for live streaming based on reinforcement learning
ABR是一种针对live streaming场景的自适应码率选择算法。本算法基于深度强化学习架构A3C，通过对用户端播放器状态进行观测以及对传输网络状况进行预测，
实时决策当前需要传输视频最合适的码率，以使得用户端在观看直播视频时可以获得最高的QoE（用户体验质量）。

## 综述
自适应码率选择算法是目前基于HTTP的视频传输中普遍采用的一类算法，旨在时变的网络状况下，为当前需要传输的视频片段实时选择合适的码率，
既能保证传输网络的带宽能够被充分利用，又能够使得用户端观看到高质量、少质量切换、低时延的视频内容，从而保证用户体验质量的高水平。
现有大部分ABR算法都是基于启发式算法，很难在多种相互矛盾的指标之间取得权衡，并且难以对网络状况做出十分准确的预测。
基于强化学习架构的自适应码率选择算法可以较好地解决传统启发式算法面临的挑战，日渐成为解决该问题的主流方案。

## 依赖环境
python==3.6
tensorflow==1.8.0
tflearn==0.3.2
numpy==1.16.5

## 运行说明
###
ABR_SDK压缩包内包含30fps的视频数据集、网络数据集、ABR算法运行代码、强化学习模型及基于C语言的算法调用代码。
###
运行代码时，需要先更改makefile中所指定的python头文件及库的路径，即将makefile文件中CFLAGS和LDFLAGS后面的路径指定为系统平台中python头文件及库所在的位置。
###
make clean，清除可执行文件及目标文件。
###
make，编译.c文件生成相应的目标文件及可执行文件。
###
生成可执行文件runpython之后，通过./runpython运行整套代码。

## 框架说明
### dataset
dataset为数据集，其中new_network_trace为网络带宽状况随时间变化的数据，包括4种网络状况：fixed、high、low、medium，每种网络状况包含20条trace；
video_trace2为30fps的视频数据，即视频每一帧的时间和对应的大小，包括4种码率的视频数据：new_frame_size_0~new_frame_size_3，分别对应2Mbps、2.5Mbps、3Mbps和3.5Mbps的视频码率。
### model
nn_model_ep_20000.ckpt.meta、nn_model_ep_20000.ckpt.index、nn_model_ep_20000.ckpt.data-00000-of-00001为深度强化学习架构训练20000 epoch后得到的模型。
### a3c3.py
自适应码率选择算法所采用的深度强化学习网络架构A3C，由actor网络和critic网络组成，网络结构有循环神经网络（GRU）及全连接神经网络。
### load_trace.py
load_trace.py用于加载网络带宽随时间变化的数据。
### ABR_v2.py
ABR_v2.py用于加载上述训练好的模型，并执行基于强化学习的码率自适应算法。输入传输网络环境及客户端播放器的相关状态：time（系统时间）、S_send_data_size（历史发送视频内容的大小，每次向网络输入历史8个GOP的视频内容的大小）、
S_time_interval（历史发送视频内容的下载时间，每次向网络输入历史8个GOP的视频内容下载时间）、S_rebuf（历史发送视频内容的再缓冲时间，即视频内容的实际播放时间与需要被播放的时间之差，每次向网络输入历史8个GOP的再缓冲时间）、
S_decision_flag（用于指示是否是需要做出视频码率决策的时间，算法中由于是每一个GOP做出一次决策，因此S_decision_flag即视频I帧标志）、end_of_video（用于指示当前视频是否结束）。
### env_v5.py
env_v5.py用于模拟网络环境，模拟视频源->CDN服务器->发送视频内容->接收视频内容的流程，并将做出码率决策后的客户端播放器及传输网络环境变化状态反馈给ABR算法。算法在系统上线后，
模拟环境的代码可以由系统真实环境所替代。
### run_v3.py
算法运行文件，通过调用ABR_v2.py中的算法运行函数run来执行自适应码率算法，同时与系统环境或者模拟环境进行交互，并计算码率决策在客户端所获得的奖励reward。
### hello.py、main.c、makefile
这3个文件为C/C++平台实现python算法调用的代码，main.c实现对指定python文件及python函数的调用，hello.py实现对算法运行文件run_v3.py的调用，makefile文件中指定python头文件路径、库路径以及编译指令。
依此执行make clean、make，生成runpython可执行文件后，通过执行./runpython实现整套代码的运行。

## 算法简介
服务器对视频进行编码，产生一帧数据后直接发送给客户端，客户端接收到I帧数据后开始实时的解码。在该系统中码率自适应模块采用深度强化学习的方法，每传输一个GOP数据后，
根据历史的传输情况，如历史的流量、卡顿情况以及接收的帧率等，决定下一个GOP选择的码率，最终实现针对直播场景实时码率自适应传输。
###
输入数据包括下载数据的大小、下载数据的时间间隔、前一个GOP的码率、视频卡顿时间、接收端接收的帧率。
###
奖励函数主要考虑当前码率、卡顿时间、码率波动3个因素，并且可以在训练过程中动态调整奖励函数中卡顿时间的惩罚因子。
###
本算法采用了深度强化学习架构A3C进行码率决策，输入为前8个时刻的历史状态值（下载数据的大小，下载数据的时间间隔，前1个GoP的码率，视频卡顿时间以及接收端接收的帧率），
actor网络负责做出自适应码率决策，其隐藏层为RNN网络，经过全连接层后再经过softmax层，输出每种码率的选择概率。critic网络负责评价决策网络所作出码率决策的效果，与actor网络具有相同的网络结构，输出层是全连接层，得到当前状态下的值函数。
actor网络每隔一个GOP进行一次码率决策，客户端会计算得到该码率下可以获得的奖励值reward，同时传输网络及客户端播放器的状态值会相应地改变。

## 实验结果
算法运行时的结果输出大致如下：
Model restored.
('network traceID, network_reward, avg_running_time', 1, 5246.996688095054, 0.0015256188132546165)
('network traceID, network_reward, avg_running_time', 2, 5282.546582605541, 0.001529967744743784)
('network traceID, network_reward, avg_running_time', 3, 5282.832038387039, 0.0015329446054066873)
('network traceID, network_reward, avg_running_time', 4, 5284.226939617484, 0.0015220555392178622)
('network traceID, network_reward, avg_running_time', 5, 5285.021053301263, 0.0015160298909402455)
('network traceID, network_reward, avg_running_time', 6, 5281.114639208133, 0.0014974226453890302)
('network traceID, network_reward, avg_running_time', 7, 5281.507385739609, 0.0014959519158308755)
('network traceID, network_reward, avg_running_time', 8, 5285.02049971161, 0.0014314626603816895)
('network traceID, network_reward, avg_running_time', 9, 5281.998843453375, 0.0014418292928625037)
('network traceID, network_reward, avg_running_time', 10, 5281.2047139962215, 0.0014881503301036078)
('network traceID, network_reward, avg_running_time', 11, 5282.581498802398, 0.0013956432792072745)
('network traceID, network_reward, avg_running_time', 12, 5279.768094082386, 0.001451501059612441)
('network traceID, network_reward, avg_running_time', 13, 5282.248584412722, 0.0014191592181170428)
('network traceID, network_reward, avg_running_time', 14, 5283.667807213661, 0.001443672822380708)
('network traceID, network_reward, avg_running_time', 15, 5283.7076645054085, 0.0014336397350837888)
('network traceID, network_reward, avg_running_time', 16, 5283.637014199148, 0.0014141350081472686)
('network traceID, network_reward, avg_running_time', 17, 5281.670782194296, 0.0014931259733257873)
('network traceID, network_reward, avg_running_time', 18, 5282.0151374053485, 0.0014613299257426149)
('network traceID, network_reward, avg_running_time', 19, 5284.674040057577, 0.0014285176691382821)
('network traceID, network_reward, avg_running_time', 0, 5280.960166231028, 0.0014677626516682533)
[5280.870008660966, 0.0014694960390277182]

## 参考文献
[1] Mao, H., Netravali, R., and Alizadeh, M. Neural adaptive video streaming with pensieve. In Proceedings of the ACM SIGCOMM 2017 Conference. ACM, 2017.

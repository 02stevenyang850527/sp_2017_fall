# sp_2017_fall
## Target:
Inspired by <https://arxiv.org/abs/1612.00694>,  plan to design a hardware LSTM engine.
## Description:
1. Implement a LSTM-based sentence generator (small model) to verify the algorithm
2. Customized a LSTM solver for framewise phone sequence recognition (TIMIT corpus) on software
## Progress:
[Note]: Simulation results below based on LSTM-based sentence generator
1. Complete the matlab code, and understand the algorithm of LSTM including both training and inference.  
2. Consider the architechure design of the solver.  
How to implement tanh and sigomid function on hardware?  
Using piecewise method  
<img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/sigmoid.png alt="sigmoid" width=300 height=250><img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/sigmoid_sim.png alt="sigmoid_sim" width=300 height=250>  
<img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/tanh.png alt="tanh" width=300 height=250><img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/tanh_pic.png alt="tanh_sim" width=300 height=250>  
Since Xavier initialization is important, how do we do random initialization?  
18 bits linear feedback shift register(need to revised: LFSR is better for zero mean)  
<img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/LFSR.png alt="LFSR" width=300 height=250><img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/lfsr_sim.png alt="LFSR_sim" width=300 height=250>  
How to calculate exponential which is part of the differential croos entropy?  
Round x and then use 2 to replace exp, since 2^x is a shift operation on hardware  
<img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/exp.png alt="exp" width=300 height=250><img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/exp_sim.png alt="exp_sim" width=300 height=250>  
How many word length should be assigned to each parameter?  
How many multipliers should we use on the architechure?  
  
## Implementation for speech recognition
1. Comparison with my code and tensorflow  
<img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/compare.png alt="compare" width=300 height=250>  
[Note]:
1. Using BasicLSTMCell and dynamic_rnn in tensorflow . 
2. My LSTM has an additional gradient clipping feature . 

## Reference
Thanks to:  
1.  <https://wiseodd.github.io/techblog/2016/08/12/lstm-backprop/>  
2.  <https://github.com/wiseodd/hipsternet>  
3.  <https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/rnn_cell_impl.py#L326>

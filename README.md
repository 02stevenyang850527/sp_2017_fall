# sp_2017_fall
## Target:
Implement LSTM solver on haradware, either FPGA or the chip
## Description:
1. Build a LSTM-based sentence generator (small model) to implement the algorithm
2. Build a bigger LSTM model for frame-based phone sequence recognition (TIMIT corpus)
## Progress:
[Note]: Simulation results below based on LSTM-based sentence generator
1. [Done] Complete the matlab code, and understand the algorithm of LSTM including both training and inference.  
2. Consider the architechure design of the solver.  
[Done] How to implement tanh and sigomid function on hardware?  
A: Using piecewise method  
<img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/sigmoid.png alt="sigmoid" width=300 height=250><img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/sigmoid_sim.png alt="sigmoid_sim" width=300 height=250>  
<img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/tanh.png alt="tanh" width=300 height=250><img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/tanh_pic.png alt="tanh_sim" width=300 height=250>  
[Done] Since Xavier initialization is important, how do we do random initialization?  
A: 18 bits linear feedback shift register(need to revised: LFSR is better for zero mean)  
<img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/LFSR.png alt="LFSR" width=300 height=250><img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/lfsr_sim.png alt="LFSR_sim" width=300 height=250>  
[Done]How to calculate exponential which is part of the differential croos entropy?  
A: round x and then use 2 to replace exp, since 2^x is a shift operation on hardware  
<img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/exp.png alt="exp" width=300 height=250><img src=https://github.com/02stevenyang850527/sp_2017_fall/blob/master/pic/exp_sim.png alt="exp_sim" width=300 height=250>  
How many word length should be assigned to each parameter?  
How many multipliers should we use on the architechure?  
  

## Reference
Thanks to:  
1.  <https://wiseodd.github.io/techblog/2016/08/12/lstm-backprop/>  
2.  <https://github.com/wiseodd/hipsternet>  
3.  <https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/rnn_cell_impl.py#L326>

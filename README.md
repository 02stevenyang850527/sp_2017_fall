# sp_2017_fall
## Target: Implement LSTM solver on haradware, either FPGA or the chip
## Description:
a LSTM-base sentence generator and a LSTM model for frame-based phone sequence recognition
## Progress:
1. [Done] Complete the matlab code, and understand the algorithm of LSTM including both training and inference.  
2. Consider the architechure design of the solver.  
[Done] How to implement tanh and sigomid function on hardware? A: Using piecewise method  
[Done] Since Xavier initialization is important, how do we do random initialization? A: 18 bits linear feedback shift register  
How many multipliers should we use on the architechure?   
How to calculate exponential which is part of the differential croos entropy?  
How many word length should be assigned to each parameter?

## Reference
Thanks to <https://github.com/wiseodd/hipsternet>

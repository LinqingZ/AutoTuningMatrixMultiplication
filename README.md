# AutoTuningMatrixMultiplication
Auto-tuning Matrix Multiplication 
- The specific implementation of auto-tuning depends on the algorithms and their adjustable parameters. Use the stored runtime information to automatically select a best fit algorithm.

### What I need to achieve:
Client interface
- [ ] input a matrix problem, which contain the row and columns of matrix, also input the matrix numbers of each row and column, and stride
- [ ] run the problem to all the algorithm and give result of all the algorithm output

profiling system
- [ ] First time when there is calculation data on the file, run the calculation for every algorithm
- [ ] When there is some data, run a AI or RL model to find the fast algorithm, and also run the matrix problem for all the algorithms and then store into data

mechanism system
- [ ] store and run quick in next time
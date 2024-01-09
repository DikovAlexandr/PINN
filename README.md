# PINN
 PINN (Physics-Informed Neural Networks) for the 2D heat equation

## Description

### Results and Issues
   - The principle of operation of PINN was studied using an example from the [video](https://www.youtube.com/watch?v=G_hIppUWcsc&ab_channel=JousefMuradLITE).
   - The basis for implementing the heat equation solver was taken from this [code](https://github.com/Samson-Mano/2D_Heat_transfer) for solving the Navier-Stokes equation and modernized to solve the two-dimensional heat equation.
   - There was an attempt to make a comparison with the solution by the finite difference method and for this purpose an analytical solution to the problem was obtained in this [notebook](OscillatorAndHeat.ipynb).
   - The analytical solution has an asymmetric appearance and apparently is not correct. Since it was not possible to find an error, the dimension of the problem under consideration was reduced to one-dimensional.
   - In the [notebook](StationaryTest.ipynb), a comparison was made of the analytical solution, the solution using the finite difference method and the solution using the PINN according to the L2 norm.
   - Now we need to understand why the solution using PINN is so different from the analytical one.
   - Possible solutions
     * Changing the optimizer - Adam produces worse results than LBFGS
     * Network architecture - 
     * Batch learning, samples - Works strangely with LBFGS
     * Analysis of setting boundary and initial conditions - There was a problem with specifying points for the equation, they were given as “for every x there is only one t” but there must be a grid
     * Weights for loss components - It was found that the boundary conditions require an increase in influence (perhaps the equation points will not be satisfied on them and they need to be excluded from the points to check the equation - this has been done)
### Research and Documentation:
   - 

### Paper Review:
   - A brief summary of the articles can be found in the [folder](research/). There are written down the main ideas that were borrowed for my code.

## References

- [2D Heat Transfer GitHub Repository](https://github.com/Samson-Mano/2D_Heat_transfer)
- [Computational Domain - YouTube Video](https://www.youtube.com/watch?v=ISp-hq6AH3Q&t=211s&ab_channel=ComputationalDomain)
- [Introduction to Heat Transfer](https://inductiva.ai/blog/article/heat-1-an-introduction)
- [Physics-Informed Neural Networks for Heat Transfer](https://inductiva.ai/blog/article/heat-2-pinn)
- [IDR LN Net Documentation](https://idrlnet.readthedocs.io/en/latest/index.html)
- [NVIDIA Modulus](https://developer.nvidia.com/modulus)

## Papers

1. [Physics-Informed Neural Networks for Solving Heat Transfer Problems](https://www.sciencedirect.com/science/article/pii/S2352179123000406#b0040)
2. [A Deep Learning Approach to Physics-Informed Neural Networks](https://epubs.siam.org/doi/epdf/10.1137/19M1274067)
3. [Introduction to Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5061501)
4. [Physics-Informed Neural Networks for Solving Partial Differential Equations](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=870037)
5. [Physics-Informed Deep Learning for Spatio-Temporal Stochastic Processes](https://www.nature.com/articles/s42254-021-00314-5)
6. [Physics-Informed Neural Networks for Turbulent Flow Modeling](https://www.nature.com/articles/s42256-021-00302-5)
7. [Physics-Informed Deep Learning for Inverse Problems](https://www.science.org/doi/full/10.1126/sciadv.abi8605)
8. [Physics-Informed Machine Learning for Material Property Prediction](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.104.025205)
9. [Physics-Informed Deep Learning for Scientific Discovery](https://iopscience.iop.org/article/10.1088/1741-4326/ab555f)
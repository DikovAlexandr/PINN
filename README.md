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
<<<<<<< HEAD
### Research and Paper Review:
   - Paper reviews will be added to the [research](research/) folder
=======
### Research and Documentation:
   - 

### Paper Review:
   - A brief summary of the articles can be found in the [folder](research/). There are written down the main ideas that were borrowed for my code.
>>>>>>> 0b74818dd9894b5aecd3ffc824d529db50a9ca77

## References
- [**SIREN**](https://github.com/vsitzmann/siren/tree/master)
- [2D Heat Transfer GitHub Repository](https://github.com/Samson-Mano/2D_Heat_transfer)
- [Computational Domain - YouTube Video](https://www.youtube.com/watch?v=ISp-hq6AH3Q&t=211s&ab_channel=ComputationalDomain)
- [Introduction to Heat Transfer](https://inductiva.ai/blog/article/heat-1-an-introduction)
- [Physics-Informed Neural Networks for Heat Transfer](https://inductiva.ai/blog/article/heat-2-pinn)
- [IDR LN Net Documentation](https://idrlnet.readthedocs.io/en/latest/index.html)
- [NVIDIA Modulus](https://developer.nvidia.com/modulus)

## Papers

1. [**DeepXDE: A Deep LearningLibrary for Solving DifferentialEquations**](https://epubs.siam.org/doi/epdf/10.1137/19M1274067)
2. [**Implicit Neural Representations with Periodic Activation Functions (SIREN)**](https://arxiv.org/abs/2006.09661)
3. [Artificial Neural Network Method for Solution of Boundary Value Problems With Exact Satisfaction of Arbitrary Boundary Conditions](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5061501)
4. [Neural-network methods for boundary value problems with irregular boundaries](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=870037)
5. [Physics-informed machine learning](https://www.nature.com/articles/s42254-021-00314-5)
6. [Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators](https://www.nature.com/articles/s42256-021-00302-5)
7. [Learning the solution operator of parametric partial differential equations with physics-informed DeepONets](https://www.science.org/doi/full/10.1126/sciadv.abi8605)
8. [Deep neural network Grad–Shafranov solver constrained with measured magnetic signals](https://iopscience.iop.org/article/10.1088/1741-4326/ab555f)
9. [Physics Informed Neural Networks towards the real-time calculation of heat fluxes at W7-X](https://www.sciencedirect.com/science/article/pii/S2352179123000406#b0040)
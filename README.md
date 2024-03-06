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
### Research and Paper Review:
   - A brief summary of the articles can be found in the [folder](docs/research/). There are written down the terminology and main ideas that were borrowed for my code.

## References
- [**SIREN**](https://github.com/vsitzmann/siren/tree/master)
- [PyTorch Adam vs LBFGS](https://github.com/youli-jlu/PyTorch_Adam_vs_LBFGS)
- [2D Heat Transfer GitHub Repository](https://github.com/Samson-Mano/2D_Heat_transfer)
- [Computational Domain - YouTube Video](https://www.youtube.com/watch?v=ISp-hq6AH3Q&t=211s&ab_channel=ComputationalDomain)
- [Introduction to Heat Transfer](https://inductiva.ai/blog/article/heat-1-an-introduction)
- [Physics-Informed Neural Networks for Heat Transfer](https://inductiva.ai/blog/article/heat-2-pinn)
- [IDR LN Net Documentation](https://idrlnet.readthedocs.io/en/latest/index.html)
- [NVIDIA Modulus](https://developer.nvidia.com/modulus)

## Papers
1. [**Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations**](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125) - 2019
2. [**Adaptive activation functions accelerate convergence in deep
and physics-informed neural networks**](https://arxiv.org/pdf/1906.01170.pdf) - 2020
3. [**Locally adaptive activation functions with slope recovery term for deep and physics-informed neural networks**](https://arxiv.org/pdf/1909.12228.pdf) - 2020
4. [**DeepXDE: A Deep LearningLibrary for Solving DifferentialEquations**](https://epubs.siam.org/doi/epdf/10.1137/19M1274067) -2021
5. [**Implicit Neural Representations with Periodic Activation Functions (SIREN)**](https://arxiv.org/abs/2006.09661) -2020
6. [**Automatic Differentiation in Machine Learning: a Survey**](https://arxiv.org/pdf/1502.05767.pdf) - 2018
7. [**On the convergence of physics informed neural networks for linear second-order elliptic and parabolic type PDEs**](https://arxiv.org/pdf/2004.01806.pdf) - 2020
8. [**Estimates on the generalization error of Physics Informed Neural Networks (PINNs) for approximating PDEs**](https://arxiv.org/pdf/2006.16144.pdf) - 2023
9. [Frequency Principle: Fourier Analysis Sheds Light On Deep Neural Networks](https://arxiv.org/pdf/1901.06523.pdf) - 2020
10. [The Old and the New: Can Physics-Informed Deep-Learning Replace Traditional Linear Solvers?](https://arxiv.org/pdf/2103.09655.pdf) - 2021
11. [Taylor-Mode Automatic Differentiation for Higher-Order Derivatives in JAX](https://openreview.net/pdf?id=SkxEF3FNPH) - 2019
12. [Characterizing possible failure modes in physics-informed neural networks](https://arxiv.org/pdf/2109.01050.pdf) - 2021
13. [Limitations of Physics Informed Machine Learning for Nonlinear Two-Phase Transport in Porous Media](https://www.researchgate.net/publication/343111185_Limitations_of_Physics_Informed_Machine_Learning_for_Nonlinear_Two-Phase_Transport_in_Porous_Media) - 2020
14. [Curriculum Learning: A Survey](https://arxiv.org/pdf/2101.10382.pdf) - 2022
15. [Artificial Neural Network Method for Solution of Boundary Value Problems With Exact Satisfaction of Arbitrary Boundary Conditions](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5061501)
16. [Neural-network methods for boundary value problems with irregular boundaries](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=870037)
17. [Physics-informed machine learning](https://www.nature.com/articles/s42254-021-00314-5)
18. [Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators](https://www.nature.com/articles/s42256-021-00302-5)
19. [Learning the solution operator of parametric partial differential equations with physics-informed DeepONets](https://www.science.org/doi/full/10.1126/sciadv.abi8605)
20. [Deep neural network Grad–Shafranov solver constrained with measured magnetic signals](https://iopscience.iop.org/article/10.1088/1741-4326/ab555f)
21. [Physics Informed Neural Networks towards the real-time calculation of heat fluxes at W7-X](https://www.sciencedirect.com/science/article/pii/S2352179123000406#b0040)
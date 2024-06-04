import os
import torch
import imageio
import numpy as np
import pandas as pd
from typing import Union
from matplotlib import cm
import matplotlib.pyplot as plt

import solver.metrics as metrics
from solver.conditions import Solution, Problem


def solution_gif(solution: Solution, 
                 frames: int, 
                 output_path='plots/animation.gif',
                 font_size=14):
    """
	Generates an animated GIF of the solution evolution over time.

	Args:
	    solution (Solution): The solution object containing the solution data.
	    frames (int): The number of frames to include in the animation.
	    output_path (str, optional): The path where the GIF will be saved (with file name).
        font_size (int, optional): The font size for the plot labels. Defaults to 14.
	"""
    x, t, u = solution.get_solution()
    times = t.unique()
    selected_times = torch.linspace(times.min(), times.max(), frames)
    print(times.shape)

    fig, ax = plt.subplots(figsize=(8, 8))
    images = []

    for time in selected_times:
        closest_time_idx = torch.argmin(torch.abs(times - time))

        if x.dim() == 1:
            # 1D problem
            ax.scatter(x[:, closest_time_idx], np.zeros_like(x[:, closest_time_idx]), 
                       c=u[:, closest_time_idx], cmap='viridis', marker='o')
            
            ax.set_title(f'Temperature at {time:.2f}', fontsize=font_size)
            ax.set_xlabel('x', fontsize=font_size)
            x_lim = (x.min(), x.max())
            ax.set_xlim(x_lim)
        elif x.dim() == 2:
            # 2D problem
            ax.scatter(x[:, 0][closest_time_idx], x[:, 1][closest_time_idx], 
                       c=u[:, closest_time_idx], cmap='viridis', marker='o')
            
            ax.set_title(f'Temperature at {time:.2f}', fontsize=font_size)
            ax.set_xlabel('x', fontsize=font_size)
            ax.set_ylabel('y', fontsize=font_size)
            x_lim = (x[:, 0].min(), x[:, 0].max())
            ax.set_xlim(x_lim)
            y_lim = (x[:, 1].min(), x[:, 1].max())
            ax.set_ylim(y_lim)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        images.append(image)
        ax.clear()

    imageio.mimsave(output_path, images, fps=10)
    plt.close(fig)


def conditions_plot(problem: Problem, 
                    t: float, 
                    output_path=None,
                    font_size=14):
    """
    Generate a plot of the initial and boundary conditions at a given time.

    Parameters:
        problem (Problem): The problem object containing the conditions data.
        t (float): The time at which to generate the plot.
        output_folder (str, optional): The folder in which to save the plot.
        font_size (int, optional): The font size for the plot labels. Defaults to 14.
    """
    x_ic, _, u_ic = problem.initial_conditions.get_initial_conditions()
    x_bc, t_bc, u_bc = problem.boundary_conditions.get_boundary_conditions()
    time_index = find_index(problem, t_bc, t).to(x_bc.device)

    plt.figure(figsize=(8, 8))

    if problem.geom.get_dimension() == 1:
        # 1D problem
        if t == 0:
            plt.scatter(to_numpy(x_ic),
                        to_numpy(u_ic),
                        marker='x', label="IC", linewidths=1)
            plt.scatter(to_numpy(x_bc[time_index]),
                        to_numpy(u_bc[time_index]),
                        marker='o', label="BC", linewidths=3)
        else:
            plt.scatter(to_numpy(x_bc[time_index]),
                        to_numpy(u_bc[time_index]),
                        marker='o', label="BC", linewidths=3)
        plt.xlim(problem.geom.limits()[0], 
                 problem.geom.limits()[1])
        plt.ylim(0, max(to_numpy(u_ic).max(),
                        to_numpy(u_bc).max()))
        plt.ylabel("u", fontsize=font_size)

    elif problem.geom.get_dimension() == 2:
        # 2D problem
        if t == 0:
            plt.scatter(to_numpy(x_ic[:, 0]),
                        to_numpy(x_ic[:, 1]),
                        c=to_numpy(u_ic), cmap='viridis',
                        marker='x', label="IC", linewidths=1)
            plt.scatter(to_numpy(x_bc[:, 0][time_index]),
                        to_numpy(x_bc[:, 1][time_index]),
                        c=to_numpy(u_bc[time_index]), cmap='viridis',
                        marker='o', label="BC", linewidths=3)
        else:
            plt.scatter(to_numpy(x_bc[:, 0][time_index]),
                        to_numpy(x_bc[:, 1][time_index]),
                        c=to_numpy(u_bc[time_index]), cmap='viridis',
                        marker='o', label="BC", linewidths=3)
        plt.xlim(problem.geom.limits()[0][0], 
                 problem.geom.limits()[0][1])
        plt.ylim(problem.geom.limits()[1][0], 
                 problem.geom.limits()[1][1])
        plt.colorbar(label='u')
        plt.ylabel("y", fontsize=font_size)

    plt.xlabel("x", fontsize=font_size)
    plt.gca().set_aspect('equal')
    plt.legend(fontsize=font_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)

    # Save the figure or display it
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def solution_surface_plot(problem: Problem, 
                          solution: Solution, 
                          output_path=None,
                          font_size=14,
                          slicer=None):
    """
    Generate a 3D surface plot of the given initial and boundary conditions and solution.

    Parameters:
        problem (Problem): The problem for which to generate the scatter plot.
        solution (Solution): The solution object containing the solution data.
        output_path (str, optional): The path to save the plot.
        font_size (int, optional): The font size for the plot labels. Defaults to 14.
        slicer (int, optional): .
    """
    x_initial, t_initial, u_initial = problem.initial_conditions.get_initial_conditions()
    slicer = int(x_initial.shape[0] / 50)
    x_initial = to_numpy(x_initial)[::slicer]
    t_initial = to_numpy(t_initial)[::slicer]
    u_initial = to_numpy(u_initial)[::slicer]

    x_boundary, t_boundary, u_boundary = problem.boundary_conditions.get_boundary_conditions()
    slicer = int(x_boundary.shape[0] / 200)
    x_boundary = to_numpy(x_boundary)[::slicer]
    t_boundary = to_numpy(t_boundary)[::slicer]
    u_boundary = to_numpy(u_boundary)[::slicer]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    if problem.geom.get_dimension() == 1:
        # 1D problem
        if solution is not None:
            x_solver, t_solver, u_solver = solution.get_solution()
            x_solver = to_numpy(x_solver)
            x_unique = np.unique(x_solver)
            t_solver = to_numpy(t_solver)
            t_unique = np.unique(t_solver)
            u_solver = to_numpy(u_solver)

            X, T = np.meshgrid(x_unique, t_unique)
            u_solver = u_solver.reshape(len(t_unique), len(x_unique)).T
            ax.plot_surface(X, T, u_solver, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        ax.scatter(x_initial, t_initial, u_initial, 
                   c='black', marker='x', label='Начальные условия')
        ax.scatter(x_boundary, t_boundary, u_boundary, 
                   c='black', marker='o', label='Граничные условия')
        ax.set_xlabel('x', fontsize=font_size+2)
        ax.set_ylabel('t', fontsize=font_size+2)

    elif problem.geom.get_dimension() == 2:
        # 2D problem
        # TODO: Show initial conditions if time is 0, add time to arguments
        # x_initial, y_initial = np.hsplit(to_numpy(x_initial), 2)
        # ax.scatter(x_initial, y_initial, to_numpy(t_initial), 
        #            c='black', marker='x', label='IC')
        if solution is not None:
            # 2D problem
            x_solver, t_solver, u_solver = solution.get_solution()
            t_solver_idx = (t_solver == t_solver.max())
            t_solver_idx = to_numpy(t_solver_idx.squeeze())
            x_solver = to_numpy(x_solver)[t_solver_idx]
            x_solver_t, y_solver_t = np.hsplit(x_solver, 2)
            x_unique_t = np.unique(x_solver_t)
            y_unique_t = np.unique(y_solver_t)
            X, Y = np.meshgrid(x_unique_t, y_unique_t, indexing='ij')
            u_solver_t = to_numpy(u_solver)[t_solver_idx].reshape(-1, 1)
            Z = u_solver_t.reshape(X.shape)

            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, 
                            linewidth=0, antialiased=False)
            
        t_boundary_idx = (t_boundary == t_boundary.max())
        t_boundary_idx = to_numpy(t_boundary_idx.squeeze())
        x_boundary = x_boundary[t_boundary_idx]
        x_boundary_t, y_boundary_t = np.hsplit(x_boundary, 2)
        u_boundary_t = u_boundary[t_boundary_idx]

        ax.scatter(x_boundary_t, y_boundary_t, u_boundary_t, 
                   c='black', marker='o', label='BC')
        ax.set_xlabel('x', fontsize=font_size+2)
        ax.set_ylabel('y', fontsize=font_size+2)

    ax.set_zlabel('u', fontsize=font_size+2)
    # ax.set_zlim(0, to_numpy(u_solver).max() if solution is not None else max(to_numpy(u_initial).max(), 
    #                                                                          to_numpy(u_boundary).max()))
    ax.set_zlim(0, max(to_numpy(u_initial).max(),to_numpy(u_boundary).max()))

    ax.legend(fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='z', which='major', labelsize=font_size)

    # Save the figure or display it
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def comparison_plot(problem: Problem,
                    solution_pinn: Solution, 
                    solution_analytical: Solution, 
                    output_folder: str, 
                    title='Comparison',
                    font_size=12):
    """
    Plots the comparison of analytical and PINN solutions.

    Parameters:
        problem (Problem): The problem for which to generate the comparison plot.
        solution_pinn (Solution): The solution object containing the PINN solution data.
        solution_analytical (Solution): The solution object containing the analytical solution data.
        output_folder (str): The path to save the plot.
        title (str, optional): The title of the plot. Defaults to 'Comparison'.
        font_size (int, optional): The font size of the plot. Defaults to 12.
    """
    plt.figure(figsize=(10, 10))
    
    if problem.geom.get_dimension() == 1:
        if solution_analytical is not None:
            plt.plot(to_numpy(solution_analytical.x), 
                    to_numpy(solution_analytical.u), 
                    label="Analytical")
        if solution_pinn is not None:
            plt.plot(to_numpy(solution_pinn.x), 
                    to_numpy(solution_pinn.u), 
                    label="PINN")
        plt.xlabel('x', fontsize=font_size)
    elif problem.geom.get_dimension() == 2:
        # TODO: make more comparable plot
        if solution_analytical is not None:
            plt.scatter(to_numpy(solution_analytical.x[:, 0]), 
                        to_numpy(solution_analytical.x[:, 1]), 
                        c=to_numpy(solution_analytical.u), 
                        marker='o', label="Analytical", cmap='viridis')
        if solution_pinn is not None:
            plt.scatter(to_numpy(solution_pinn.x[:, 0]), 
                        to_numpy(solution_pinn.x[:, 1]), 
                        c=to_numpy(solution_pinn.u), 
                        marker='x', label="PINN", cmap='viridis')
        plt.colorbar(label='u')
    plt.ylabel('u', fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    if output_folder:
        plt.savefig(os.path.join(output_folder, title + '.png'))
        plt.show()
    else:
        plt.show()


def loss_history_plot(data_path: str, 
                      output_folder: str, 
                      is_log=False, 
                      title='LossHistory',
                      font_size=12):
    """
    Generate a plot of loss history from the data.

    Parameters:
        data_path (str): The path to the CSV file containing the loss history data.
        output_folder (str): The folder where the plot will be saved.
        is_log (bool, optional): If True, plot the data using a logarithmic scale.
        title (str, optional): The title of the plot.
    """
    if os.path.exists(data_path):
        loss_history = pd.read_csv(data_path, header=None)
        if is_log:
            plt.semilogy(loss_history[0], loss_history[1])
            plt.ylabel('Logarithmic Loss', fontsize=font_size)
        else:
            plt.plot(loss_history[0], loss_history[1])
            plt.ylabel('Loss', fontsize=font_size)
        plt.xlabel('Epoch', fontsize=font_size)
        plt.title(title, fontsize=font_size)
        plt.savefig(os.path.join(output_folder, title + '.png'))
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        plt.show()


def error_plot(problem: Problem,
               solution_pinn: Solution, 
               solution_analytical: Solution, 
               output_path=None,
               font_size=12):
    """
    Generate a plot of the error between the solution and the true solution.

    Parameters:
        problem (Problem): The problem for which to generate the error plot.
        solution_pinn (Solution): The solution object containing the PINN solution data.
        solution_analytical (Solution): The solution object containing the analytical solution data.
        output_path (str, optional): The path to save the plot.
        font_size (int, optional): The font size of the plot. Defaults to 12.
    """
    x_pinn, _, u_pinn = solution_pinn.get_solution()
    x_analytical, _, u_analytical = solution_analytical.get_solution()

    error = metrics.calculate_error(u_analytical, u_pinn, is_abs=False)
    
    plt.figure(figsize=(8, 8))

    if problem.geom.get_dimension() == 1:
        # 1D problem
        plt.plot(to_numpy(x_pinn), to_numpy(error), label='Error')
        plt.ylabel('Error', fontsize=font_size)
        
    elif problem.geom.get_dimension() == 2:
        # 2D problem
        plt.scatter(to_numpy(x_pinn[:, 0]), to_numpy(x_pinn[:, 1]), to_numpy(error), 
                    cmap='viridis', cbar=True)
        plt.ylabel('y', fontsize=font_size)

    plt.xlabel('x', fontsize=font_size)
    plt.title('Error Plot', fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)

    # Save the figure or display it
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def evolution_gif(problem: Problem,
                  solution: Solution, 
                  frames: int, 
                  output_path='plots/animation3d.gif',
                  font_size=12):
    """
    Generate a 3D GIF animation of the evolution of a solution over time.
    ONLY FOR 1D PROBLEMS.

    Parameters:
        problem (Problem): The problem for which to generate the animation.
        solution (Solution): The solution to the problem.
        frames (int): The number of frames in the animation.
        output_path (str, optional): The path to save the animation.
        font_size (int, optional): The font size of the plot. Defaults to 12.
    """
    x, t, u = solution.get_solution()

    images = []

    x_lim = (x.min(), x.max())
    t_lim = (t.min(), t.max())
    u_lim = (u.min(), u.max())
    u_range = np.linspace(u.min(), u.max(), frames)
    
    # Needed for slice surface
    dense_x = np.linspace(x.min(), x.max(), 10)
    dense_t = np.linspace(t.min(), t.max(), 10)
    xx, tt = np.meshgrid(dense_x, dense_t)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, u_slice in enumerate(u_range):
        under_plane = u < u_slice

        if problem.geom.get_dimension() == 1:
            ax.scatter(x[under_plane], t[under_plane], u[under_plane], 
                    c=u[under_plane], marker='o')
            
            ax.plot_surface(xx, tt, np.full_like(xx, u_slice), alpha=0.2)

            ax.set_xlim(x_lim)
            ax.set_ylim(t_lim)
            ax.set_zlim(u_lim)

            ax.set_xlabel('x', fontsize=font_size)
            ax.set_ylabel('t', fontsize=font_size)
            ax.set_zlabel('u', fontsize=font_size)

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            images.append(image)
            ax.clear()

        elif problem.geom.get_dimension() == 2:
            # TODO: Implement for 2D
            pass

    imageio.mimsave(output_path, images, fps=10)
    plt.close(fig)


def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert the input tensor to a NumPy array. If the input is already a NumPy array, return it unchanged.

    Parameters:
        tensor (torch.Tensor or numpy.ndarray): Input tensor to be converted.

    Returns:
        numpy.ndarray: Converted NumPy array.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    else:
        raise TypeError("Input must be a torch.Tensor or a numpy.ndarray")


def find_index(problem: Problem, 
               tensor: torch.Tensor, 
               value: float) -> torch.Tensor:
    """
    Find the index of a value in a tensor that is close to the given value.

    Parameters:
        problem (object): The problem object.
        tensor (torch.Tensor): The tensor to search in.
        value (float): The value to find the index of.

    Returns:
        torch.Tensor: A tensor containing the indices of the values in the tensor that are close to the given value.
    """
    # Epsilon is half the grid spacing
    return torch.where(torch.abs(tensor - value) <= 0.5 * problem.period.grid_spacing_inners())[0] 
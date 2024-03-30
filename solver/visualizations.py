import imageio
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch

import solver.metrics as metrics

def solution_gif(solution, frames, output_path='plots/animation.gif'):
    """
	Generates an animated GIF of the solution evolution over time.

	Args:
	    solution: The solution object containing the solution data.
	    frames: The number of frames to include in the animation.
	    output_path: The path where the GIF will be saved.
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
            
            ax.set_title(f'Temperature at {time:.2f}')
            ax.set_xlabel('x')
            x_lim = (x.min(), x.max())
            ax.set_xlim(x_lim)
        elif x.dim() == 2:
            # 2D problem
            ax.scatter(x[:, 0][closest_time_idx], x[:, 1][closest_time_idx], 
                       c=u[:, closest_time_idx], cmap='viridis', marker='o')
            
            ax.set_title(f'Temperature at {time:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
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


def conditions_plot(problem, t, output_path=None):
    """
    Generate a plot of the initial and boundary conditions at a given time for the given problem.

    Parameters:
        problem: The problem object containing the conditions data.
        t: The time at which to generate the plot.
        output_folder: The folder in which to save the plot. If not provided, the plot is displayed instead.
    """
    x_ic, _, u_ic = problem.initial_conditions.get_initial_conditions()
    x_bc, t_bc, u_bc = problem.boundary_conditions.get_boundary_conditions()
    time_index = find_index(problem, t_bc, t)

    plt.figure(figsize=(8, 8))

    if x_ic.dim() == 1:
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
        plt.xlim(problem.geom.limits()[0], problem.geom.limits()[1])
        plt.ylim(0, max(to_numpy(u_ic).max(),
                        to_numpy(u_bc).max()))
        plt.xlabel("x")
        plt.ylabel("u")
    elif x_ic.dim() == 2:
        # 2D problem
        if t == 0:
            plt.scatter(to_numpy(x_ic[:, 0]),
                        to_numpy(x_ic[:, 1]),
                        c=u_ic, cmap='viridis',
                        marker='x', label="IC", linewidths=1)
            plt.scatter(to_numpy(x_bc[:, 0][time_index]),
                        to_numpy(x_bc[:, 1][time_index]),
                        c=u_bc[time_index], cmap='viridis',
                        marker='o', label="BC", linewidths=3)
        else:
            plt.scatter(to_numpy(x_bc[:, 0][time_index]),
                        to_numpy(x_bc[:, 1][time_index]),
                        c=u_bc[time_index], cmap='viridis',
                        marker='o', label="BC", linewidths=3)
        plt.xlim(problem.geom.limits()[0][0], 
                 problem.geom.limits()[0][1])
        plt.ylim(problem.geom.limits()[1][0], 
                 problem.geom.limits()[1][1])
        plt.colorbar(label='u')
        plt.xlabel("x")
        plt.ylabel("y")
    plt.legend()

    # Save the figure or display it
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def solution_surface_plot(problem, solution, output_path=None):
    """
    Generate a 3D surface plot of the given initial and boundary conditions and solution. 
    Parameters:
        problem: The problem for which to generate the scatter plot.
        solution: The solution to the problem.
        output_path: The path to save the plot (optional).
    """
    x_initial, t_initial, u_initial = problem.initial_conditions.get_initial_conditions()
    x_boundary, t_boundary, u_boundary = problem.boundary_conditions.get_boundary_conditions()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(to_numpy(x_initial), to_numpy(t_initial), to_numpy(u_initial), 
               c='black', marker='x', label='IC')
    ax.scatter(to_numpy(x_boundary), to_numpy(t_boundary), to_numpy(u_boundary), 
               c='black', marker='o', label='BC')

    if solution is not None:
        x_solver, t_solver, u_solver = solution.get_solution()
        ax.plot_surface(to_numpy(x_solver), to_numpy(t_solver), to_numpy(u_solver), 
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    ax.legend()

    # Save the figure or display it
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_error(solution, y_true, output_path=None):
    x_solver, _, u_solver = solution.get_solution()
    error = metrics.calculate_error(y_true, u_solver, is_abs=False)
    
    plt.figure(figsize=(8, 8))

    if x_solver.dim() == 1:
        # 1D problem
        plt.plot(to_numpy(x_solver), to_numpy(error), label='Error')
        plt.xlabel('x')
        plt.ylabel('Error')
        
    elif x_solver.dim() == 2:
        # 2D problem
        plt.scatter(to_numpy(x_solver[:, 0]), to_numpy(x_solver[:, 1]), to_numpy(error), 
                    cmap='viridis', cbar=True)
        plt.xlabel('x')
        plt.ylabel('y')

    plt.title('Error Plot')
    plt.legend()

    # Save the figure or display it
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def evolution_gif(solution, frames, output_path='plots/animation.gif'):
    """
    Generate a GIF animation of the evolution of a solution over time.
    ONLY FOR 1D PROBLEMS.

    Parameters:
    - solution: the solution object containing the data to be visualized.
    - frames: the number of frames in the animation.
    - output_path: the file path to save the animation.
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

        if x.dim() == 1:
            ax.scatter(x[under_plane], t[under_plane], u[under_plane], 
                    c=u[under_plane], marker='o')
            
            ax.plot_surface(xx, tt, np.full_like(xx, u_slice), alpha=0.2)

            ax.set_xlim(x_lim)
            ax.set_ylim(t_lim)
            ax.set_zlim(u_lim)

            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_zlabel('u')

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            images.append(image)
            ax.clear()

    imageio.mimsave(output_path, images, fps=10)
    plt.close(fig)

def to_numpy(tensor):
    """
    Convert the input tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): Input tensor to be converted.

    Returns:
        numpy.ndarray: Converted NumPy array.
    """
    return tensor.cpu().detach().numpy()

def find_index(problem, tensor, value):
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
    return torch.where(torch.abs(tensor - value) <= 0.5 * problem.time.grid_spacing_inners())[0] 
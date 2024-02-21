import numpy as np


class LossWeightAdjuster:
    def __init__(self, max_weight, min_weight, threshold, scaling_factor):
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.threshold = threshold
        self.scaling_factor = scaling_factor

    def adjust_weights(self, weights, losses):
        for i in range(len(weights)):
            if losses[i] < self.threshold:
                weights[i] = weights[i] / self.scaling_factor
            else:
                weights[i] = losses[i] * self.scaling_factor
            weights[i] = max(self.min_weight, min(weights[i], self.max_weight))


def rar_points(geom, period, X, T, errors, num_points, epsilon, random=True):
    max_index = np.argmax(np.absolute(errors))
    center_coords = [X[max_index], T[max_index]]
    dimension = len(center_coords[0])

    if random:
        x_extra = center_coords[0] + np.random.normal(0, epsilon, size=(num_points, dimension))
        t_extra = center_coords[1] + np.random.normal(0, epsilon, num_points)
    else:
        n = int((num_points ** (1/(1+dimension))) / 2)
        x_extra = []
        for i in range(dimension):
            x_i = center_coords[0][i] + np.linspace(-geom.grid_spacing_inners()[i] * n,
                                                    geom.grid_spacing_inners()[i] * n, 2 * n + 1)
            x_extra.append(x_i)
        x_extra = np.column_stack(x_extra)
        t_extra = center_coords[1] + np.linspace(-period.grid_spacing_inners() * n,
                                         period.grid_spacing_inners() * n, 2*n + 1)
        # Make a grid
        if dimension == 1:
            x_extra, t_extra = np.meshgrid(x_extra,  t_extra)
            x_extra = x_extra.flatten()
            t_extra = t_extra.flatten()
        elif dimension == 2:
            x_extra, y_extra, t_extra = np.meshgrid(x_extra[:, 0],  x_extra[:, 1], t_extra)
            x_extra = np.column_stack((x_extra.flatten(), y_extra.flatten()))
            t_extra = t_extra.flatten()
    
    # Clip offsets to the boundaries
    new_points = []
    for x, t in zip(x_extra, t_extra):
        new_points.append([x.tolist(), t])
    new_points = [point for point in new_points if geom.inside(point[0]) and period.inside(point[1])]
    x = np.array([item[0] for item in new_points])
    t = np.array([item[1] for item in new_points])
    return x, t, (center_coords[0], center_coords[1])
import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Vendored PINA (pinnacle/pina)
from pina import LabelTensor, Trainer
from pina.callback import MetricTracker
from pina.model import FeedForward
from pina.problem.zoo import (
    AcousticWaveProblem,
    AdvectionProblem,
    AllenCahnProblem,
    DiffusionReactionProblem,
    HelmholtzProblem,
    Poisson2DSquareProblem,
    SupervisedProblem,
)
from pina.solver import (
    CausalPINN,
    CompetitivePINN,
    DeepEnsemblePINN,
    GradientPINN,
    PINN,
    RBAPINN,
    SelfAdaptivePINN,
    SupervisedSolver,
)


@dataclass(frozen=True)
class PinaCase:
    name: str
    build_problem: Callable[[], object]


@dataclass(frozen=True)
class RefCase:
    name: str
    ref_name: str
    input_vars: List[str]
    output_vars: List[str]
    timepde: Optional[Tuple[float, float]] = None
    downsample: Optional[int] = None


REF_CASES: Dict[str, RefCase] = {
    "Burgers1D": RefCase(
        name="Burgers1D",
        ref_name="burgers1d",
        input_vars=["x", "t"],
        output_vars=["u"],
        timepde=(0.0, 1.0),
    ),
    "Burgers2D": RefCase(
        name="Burgers2D",
        ref_name="burgers2d",
        input_vars=["x", "y", "t"],
        output_vars=["u", "v"],
        timepde=(0.0, 1.0),
    ),
    "Poisson2D_Classic": RefCase(
        name="Poisson2D_Classic",
        ref_name="poisson1_cg_data",
        input_vars=["x", "y"],
        output_vars=["u"],
    ),
    "Poisson2DManyArea": RefCase(
        name="Poisson2DManyArea",
        ref_name="poisson_manyarea",
        input_vars=["x", "y"],
        output_vars=["u"],
    ),
    "Poisson2DBoltzmann": RefCase(
        name="Poisson2DBoltzmann",
        ref_name="poisson_boltzmann2d",
        input_vars=["x", "y"],
        output_vars=["u"],
    ),
    "Poisson3D": RefCase(
        name="Poisson3D",
        ref_name="poisson_3d",
        input_vars=["x", "y", "z"],
        output_vars=["u"],
    ),
    "HeatMultiscale": RefCase(
        name="HeatMultiscale",
        ref_name="heat_multiscale_lesspoints",
        input_vars=["x", "y", "t"],
        output_vars=["u"],
        timepde=(0.0, 5.0),
    ),
    "HeatComplex": RefCase(
        name="HeatComplex",
        ref_name="heat_complex",
        input_vars=["x", "y", "t"],
        output_vars=["u"],
        timepde=(0.0, 3.0),
    ),
    "HeatLongTime": RefCase(
        name="HeatLongTime",
        ref_name="heat_longtime",
        input_vars=["x", "y", "t"],
        output_vars=["u"],
        timepde=(0.0, 100.0),
    ),
    "HeatDarcy": RefCase(
        name="HeatDarcy",
        ref_name="heat_darcy",
        input_vars=["x", "y", "t"],
        output_vars=["u"],
        timepde=(0.0, 5.0),
    ),
    "WaveHetergeneous": RefCase(
        name="WaveHetergeneous",
        ref_name="wave_darcy",
        input_vars=["x", "y", "t"],
        output_vars=["u"],
        timepde=(0.0, 5.0),
    ),
    "GrayScott": RefCase(
        name="GrayScott",
        ref_name="grayscott",
        input_vars=["x", "y", "t"],
        output_vars=["u", "v"],
        downsample=6,
    ),
    "Kuramoto": RefCase(
        name="Kuramoto",
        ref_name="Kuramoto_Sivashinsky",
        input_vars=["x", "t"],
        output_vars=["u"],
        downsample=6,
    ),
    "NS_Long": RefCase(
        name="NS_Long",
        ref_name="ns_long",
        input_vars=["x", "y", "t"],
        output_vars=["u", "v", "p"],
        timepde=(0.0, 5.0),
        downsample=6,
    ),
    "NS_FourCircles": RefCase(
        name="NS_FourCircles",
        ref_name="ns_4_obstacle",
        input_vars=["x", "y"],
        output_vars=["u", "v", "p"],
    ),
    "NS_NoObstacle": RefCase(
        name="NS_NoObstacle",
        ref_name="ns_0_obstacle",
        input_vars=["x", "y"],
        output_vars=["u", "v", "p"],
    ),
}


def _set_seed(seed: Optional[int]) -> int:
    if seed is None:
        seed = random.randint(0, 10**9)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def _device_from_arg(device: str) -> Tuple[str, Dict[str, object]]:
    """Map `--device` like pinnacle uses ('cpu' or '0') to Lightning args."""
    if device == "cpu":
        return "cpu", {"accelerator": "cpu", "devices": 1}
    # GPU index string, e.g. "0"
    return "cuda", {"accelerator": "gpu", "devices": [int(device)]}


def _build_feedforward(problem, hidden_layers: List[int]) -> torch.nn.Module:
    input_dim = len(problem.input_variables)
    output_dim = len(problem.output_variables)
    return FeedForward(
        input_dimensions=input_dim,
        output_dimensions=output_dim,
        layers=hidden_layers,
    )


def _build_solver(method: str, problem, model: torch.nn.Module, ensemble_n: int):
    if method == "pina_pinn":
        return PINN(problem=problem, model=model)
    if method == "pina_gradient":
        return GradientPINN(problem=problem, model=model)
    if method == "pina_causal":
        return CausalPINN(problem=problem, model=model)
    if method == "pina_competitive":
        return CompetitivePINN(problem=problem, model=model)
    if method == "pina_selfadaptive":
        return SelfAdaptivePINN(problem=problem, model=model)
    if method == "pina_rba":
        return RBAPINN(problem=problem, model=model)
    if method == "pina_ensemble":
        models = [_build_feedforward(problem, [64, 64, 64]) for _ in range(ensemble_n)]
        return DeepEnsemblePINN(problem=problem, models=models)
    if method == "pina_supervised":
        return SupervisedSolver(problem=problem, model=model)
    raise ValueError(f"Unsupported PINA method: {method}")


def _parse_hidden_layers(hidden: str) -> List[int]:
    if not hidden:
        return [64, 64, 64]
    return [int(x.strip()) for x in hidden.split(",") if x.strip()]


def _resolve_ref_path(ref_dir: str, ref_name: str) -> str:
    if os.path.isabs(ref_dir):
        return os.path.join(ref_dir, f"{ref_name}.dat")
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, ref_dir, f"{ref_name}.dat")


def _load_ref_data(ref_dir: str, case: RefCase) -> Tuple[np.ndarray, np.ndarray]:
    input_dim = len(case.input_vars)
    output_dim = len(case.output_vars)
    ref_path = _resolve_ref_path(ref_dir, case.ref_name)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference data file not found: {ref_path}")

    with open(ref_path, "r", encoding="utf-8", errors="ignore") as f:
        data = np.loadtxt(f, comments="%").astype(np.float32)
    if case.timepde is not None:
        t_start, t_end = case.timepde
        num_tsample = (data.shape[1] - (input_dim - 1)) // output_dim
        if num_tsample <= 0 or num_tsample * output_dim != data.shape[1] - (input_dim - 1):
            raise ValueError(
                f"Invalid timepde layout for {case.name}: "
                f"cols={data.shape[1]}, input_dim={input_dim}, output_dim={output_dim}"
            )

        t = np.linspace(t_start, t_end, num_tsample, dtype=np.float32)
        t_mesh, x0 = np.meshgrid(t, data[:, 0].astype(np.float32))
        list_cols = [x0.reshape(-1)]
        for i in range(1, input_dim - 1):
            list_cols.append(
                np.stack([data[:, i] for _ in range(num_tsample)], axis=1).reshape(-1)
            )
        list_cols.append(t_mesh.reshape(-1))
        for i in range(output_dim):
            list_cols.append(
                data[:, input_dim - 1 + i :: output_dim].reshape(-1)
            )
        data = np.stack(list_cols, axis=1).astype(np.float32)

    ref_x = data[:, :input_dim]
    ref_y = data[:, input_dim : input_dim + output_dim]
    return ref_x, ref_y


def _downsample_ref_data(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    factor: Optional[int],
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if not factor or factor <= 1:
        return ref_x, ref_y
    rng = np.random.default_rng(seed)
    ndat = ref_x.shape[0]
    take = max(1, ndat // factor)
    idx = rng.choice(np.arange(ndat), size=take, replace=False)
    return ref_x[idx], ref_y[idx]


def _split_ref_data(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    n_train: int,
    n_eval: int,
    seed: Optional[int],
    shuffle: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ndat = ref_x.shape[0]
    idx = np.arange(ndat)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    ref_x = ref_x[idx]
    ref_y = ref_y[idx]

    if n_train <= 0:
        n_train = ndat
    n_train = min(n_train, ndat)
    if n_eval <= 0:
        n_eval = max(0, ndat - n_train)
    n_eval = min(n_eval, ndat - n_train)

    train_x = ref_x[:n_train]
    train_y = ref_y[:n_train]
    eval_x = ref_x[n_train : n_train + n_eval]
    eval_y = ref_y[n_train : n_train + n_eval]
    return train_x, train_y, eval_x, eval_y


@torch.no_grad()
def _evaluate_case(solver, problem, eval_n: int) -> Dict[str, float]:
    """Compute metrics compatible with `src/utils/summary.py` errors.txt parsing."""
    if not hasattr(problem, "solution"):
        return {"mae": float("nan"), "mse": float("nan"), "mxe": float("nan"), "l1re": float("nan"), "l2re": float("nan")}

    # Prefer random points to avoid grid explosion in space-time domains
    pts = problem.domains["D"].sample(eval_n, "random").sort_labels()

    y_true = problem.solution(pts)
    y_pred = solver(pts)

    # y_true and y_pred can be LabelTensor; normalize to plain tensors
    if hasattr(y_true, "tensor"):
        y_true_t = y_true.tensor
    else:
        y_true_t = torch.as_tensor(y_true)
    if hasattr(y_pred, "tensor"):
        y_pred_t = y_pred.tensor
    else:
        y_pred_t = torch.as_tensor(y_pred)

    # Flatten
    y_true_t = y_true_t.reshape(-1)
    y_pred_t = y_pred_t.reshape(-1)
    diff = y_true_t - y_pred_t

    mae = torch.mean(torch.abs(diff)).item()
    mse = torch.mean(diff**2).item()
    mxe = torch.max(torch.abs(diff)).item()
    l1re = (torch.norm(diff, p=1) / torch.norm(y_true_t, p=1)).item()
    l2re = (torch.norm(diff, p=2) / torch.norm(y_true_t, p=2)).item()

    return {
        "mae": mae,
        "mse": mse,
        "mxe": mxe,
        "l1re": l1re,
        "l2re": l2re,
    }


@torch.no_grad()
def _evaluate_ref_case(
    solver,
    eval_inputs: LabelTensor,
    eval_targets: torch.Tensor,
) -> Dict[str, float]:
    if eval_inputs.numel() == 0:
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "mxe": float("nan"),
            "l1re": float("nan"),
            "l2re": float("nan"),
        }

    device = next(solver.model.parameters()).device
    eval_inputs = LabelTensor(eval_inputs.to(device), labels=eval_inputs.labels)
    y_true_t = eval_targets.to(device)
    y_pred = solver(eval_inputs)
    y_pred_t = y_pred.tensor if hasattr(y_pred, "tensor") else torch.as_tensor(y_pred)

    y_true_t = y_true_t.reshape(-1)
    y_pred_t = y_pred_t.reshape(-1)
    diff = y_true_t - y_pred_t

    mae = torch.mean(torch.abs(diff)).item()
    mse = torch.mean(diff**2).item()
    mxe = torch.max(torch.abs(diff)).item()
    l1re = (torch.norm(diff, p=1) / torch.norm(y_true_t, p=1)).item()
    l2re = (torch.norm(diff, p=2) / torch.norm(y_true_t, p=2)).item()

    return {
        "mae": mae,
        "mse": mse,
        "mxe": mxe,
        "l1re": l1re,
        "l2re": l2re,
    }


def _generate_plots(exp_path: str, tasknum: int, column_width: str) -> None:
    try:
        from src.visualization.compare import load_experiment_results
        from fbpinns.plot.style import apply_ieee_style, ieee_figsize
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"Warning: plot tools not available: {exc}")
        return

    try:
        results = load_experiment_results(exp_path)
    except Exception as exc:
        print(f"Warning: could not load results for plots: {exc}")
        return

    def plot_statistical_histories_fbpinns(histories_per_experiment, metric, output_path):
        apply_ieee_style()
        fig_w, fig_h = ieee_figsize(column_width, aspect=0.6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        max_steps = 0
        for trial_histories in histories_per_experiment.values():
            for h in trial_histories:
                x_key = "steps" if "steps" in h else "epochs"
                if x_key in h and len(h[x_key]) > 0:
                    max_steps = max(max_steps, h[x_key][-1])

        if max_steps == 0:
            raise ValueError("No valid data found in histories")

        common_x_axis = np.linspace(0, max_steps, 200)
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])

        for idx, (name, trial_histories) in enumerate(histories_per_experiment.items()):
            interpolated_values = []
            for h in trial_histories:
                x_key = "steps" if "steps" in h else "epochs"
                if x_key not in h or metric not in h:
                    continue
                x_data = h[x_key]
                y_data = h[metric]
                if len(x_data) == 0 or len(y_data) == 0:
                    continue
                valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                x_data = x_data[valid_mask]
                y_data = y_data[valid_mask]
                if len(x_data) < 2:
                    continue
                interp_y = np.interp(common_x_axis, x_data, y_data, left=y_data[0], right=y_data[-1])
                interpolated_values.append(interp_y)

            if len(interpolated_values) == 0:
                print(f"Warning: No valid data for {name}")
                continue

            interpolated_values = np.array(interpolated_values)
            mean_curve = np.mean(interpolated_values, axis=0)
            std_curve = np.std(interpolated_values, axis=0)

            color = colors[idx % len(colors)]
            ax.semilogy(common_x_axis, mean_curve, label=name, color=color, linewidth=1.2)
            lower_bound = np.maximum(mean_curve - std_curve, 1e-10)
            upper_bound = mean_curve + std_curve
            ax.fill_between(common_x_axis, lower_bound, upper_bound, color=color, alpha=0.2)

        metric_labels = {
            "train": "Training Loss",
            "test": "Test Loss",
            "l2re": "L2 Relative Error",
            "mse": "Mean Square Error",
            "mae": "Mean Absolute Error",
            "mxe": "Maximum Error",
        }
        ax.set_xlabel("Steps")
        ax.set_ylabel(metric_labels.get(metric, metric.upper()))
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, which="major", linestyle="--", alpha=0.25, linewidth=0.5, color="gray")
        ax.grid(False, which="minor")
        ax.minorticks_on()

        plt.tight_layout()
        if output_path is not None:
            fig.savefig(output_path)
        plt.close(fig)

    figs_root = os.path.join(exp_path, "figs")
    os.makedirs(figs_root, exist_ok=True)

    for task_id in range(tasknum):
        task_key = f"task-{task_id}"
        task_dir = os.path.join(figs_root, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)

        if task_key in results.get("loss", {}):
            histories = {"PINA": results["loss"][task_key]}
            try:
                plot_statistical_histories_fbpinns(
                    histories,
                    metric="train",
                    output_path=os.path.join(task_dir, "train_loss.pdf"),
                )
            except Exception as exc:
                print(f"Warning: plot train loss failed for {task_key}: {exc}")

        if task_key in results.get("metrics", {}):
            histories = {"PINA": results["metrics"][task_key]}
            for metric in ("l2re", "mse"):
                try:
                    plot_statistical_histories_fbpinns(
                        histories,
                        metric=metric,
                        output_path=os.path.join(task_dir, f"{metric}.pdf"),
                    )
                except Exception as exc:
                    print(f"Warning: plot {metric} failed for {task_key}: {exc}")


def _write_loss(path: str, steps: np.ndarray, train_losses: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        # loss.txt is expected to have: step + train/test/weight components.
        # Use a single component with deterministic numeric values so plots work.
        f.write("# step, loss_train, loss_test, loss_weight\n")
        for step, loss in zip(steps, train_losses):
            f.write(
                f"{float(step):.18e} "
                f"{float(loss):.18e} "
                f"{float(loss):.18e} "
                f"{float(1.0):.18e}\n"
            )


def _write_errors(path: str, steps: np.ndarray, metrics_seq: List[Dict[str, float]]) -> None:
    # Must match: epoch, mae, mse, mxe, l1re, l2re, crmse, frmses(low, mid, high)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# epochs, maes, mses, mxes, l1res, l2res, crmses, frmses(low, mid, high)\n")
        for step, metrics in zip(steps, metrics_seq):
            f.write(
                f"{float(step):.18e} "
                f"{metrics['mae']:.18e} "
                f"{metrics['mse']:.18e} "
                f"{metrics['mxe']:.18e} "
                f"{metrics['l1re']:.18e} "
                f"{metrics['l2re']:.18e} "
                f"{float('nan'):.18e} "
                f"{float('nan'):.18e} {float('nan'):.18e} {float('nan'):.18e}\n"
            )


def _labeltensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "tensor"):
        value = value.tensor
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _extract_train_loss_history(callbacks: List[object]) -> List[float]:
    for cb in callbacks:
        if isinstance(cb, MetricTracker):
            metrics = cb.metrics
            if "train_loss" in metrics:
                return metrics["train_loss"].detach().cpu().numpy().tolist()
            if "train_loss_epoch" in metrics:
                return metrics["train_loss_epoch"].detach().cpu().numpy().tolist()
    return []


def _evaluate_case_on_points(solver, problem, pts) -> Dict[str, float]:
    if not hasattr(problem, "solution"):
        return {"mae": float("nan"), "mse": float("nan"), "mxe": float("nan"), "l1re": float("nan"), "l2re": float("nan")}

    y_true = problem.solution(pts)
    y_pred = solver(pts)

    y_true_t = y_true.tensor if hasattr(y_true, "tensor") else torch.as_tensor(y_true)
    y_pred_t = y_pred.tensor if hasattr(y_pred, "tensor") else torch.as_tensor(y_pred)

    y_true_t = y_true_t.reshape(-1)
    y_pred_t = y_pred_t.reshape(-1)
    diff = y_true_t - y_pred_t

    mae = torch.mean(torch.abs(diff)).item()
    mse = torch.mean(diff**2).item()
    mxe = torch.max(torch.abs(diff)).item()
    l1re = (torch.norm(diff, p=1) / torch.norm(y_true_t, p=1)).item()
    l2re = (torch.norm(diff, p=2) / torch.norm(y_true_t, p=2)).item()

    return {"mae": mae, "mse": mse, "mxe": mxe, "l1re": l1re, "l2re": l2re}


def _select_slice_for_plot(
    inputs: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if inputs.shape[1] <= 2:
        return inputs, y_true, y_pred
    # Slice by last dimension (often time); keep closest points to mid value.
    last_dim = inputs[:, -1]
    mid_val = 0.5 * (np.min(last_dim) + np.max(last_dim))
    idx = np.argsort(np.abs(last_dim - mid_val))[:max_points]
    return inputs[idx, :2], y_true[idx], y_pred[idx]


def _plot_scalar_field(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    title: str,
    out_path: str,
    cmap=None,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    plt.figure()
    triang = mtri.Triangulation(x, y)
    plt.tricontourf(triang, values, levels=32, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_solution_outputs(
    exp_path: str,
    task_id: int,
    repeat_id: int,
    input_vars: List[str],
    output_vars: List[str],
    inputs: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    column_width: str = "single",
) -> None:
    try:
        import matplotlib.pyplot as plt
        from fbpinns.plot.style import apply_ieee_style, ieee_figsize
        from fbpinns.plot import palette
        import matplotlib.tri as mtri
    except Exception as exc:
        print(f"Warning: matplotlib not available for solution plots: {exc}")
        return

    apply_ieee_style()

    figs_root = os.path.join(exp_path, "figs")
    task_dir = os.path.join(figs_root, f"task_{task_id}")
    os.makedirs(task_dir, exist_ok=True)

    inputs, y_true, y_pred = _select_slice_for_plot(inputs, y_true, y_pred)
    x_dim = inputs.shape[1]
    y_dim = y_true.shape[1] if y_true.ndim > 1 else 1
    y_true = y_true.reshape(-1, y_dim)
    y_pred = y_pred.reshape(-1, y_dim)

    if x_dim == 1:
        sort_idx = np.argsort(inputs[:, 0])
        xs = inputs[sort_idx, 0]
        for i in range(y_dim):
            fig_w, fig_h = ieee_figsize(column_width, aspect=0.6)
            plt.figure(figsize=(fig_w, fig_h))
            plt.plot(xs, y_true[sort_idx, i], label="reference")
            plt.plot(xs, y_pred[sort_idx, i], "--", label="prediction")
            plt.legend()
            plt.xlabel(input_vars[0] if input_vars else "x")
            plt.ylabel(output_vars[i] if output_vars else f"y{i}")
            plt.tight_layout()
            plt.savefig(os.path.join(task_dir, f"{output_vars[i]}_solution_r{repeat_id}.pdf"))
            plt.close()

            plt.figure(figsize=(fig_w, fig_h))
            plt.plot(xs, y_true[sort_idx, i], label="reference")
            plt.plot(xs, y_pred[sort_idx, i], "--", label="prediction")
            plt.plot(xs, y_pred[sort_idx, i] - y_true[sort_idx, i], label="error")
            plt.legend()
            plt.xlabel(input_vars[0] if input_vars else "x")
            plt.ylabel(output_vars[i] if output_vars else f"y{i}")
            plt.tight_layout()
            plt.savefig(os.path.join(task_dir, f"{output_vars[i]}_triptych_r{repeat_id}.pdf"))
            plt.close()

            plt.figure()
            plt.plot(xs, y_pred[sort_idx, i] - y_true[sort_idx, i], label="prediction - reference")
            plt.xlabel(input_vars[0] if input_vars else "x")
            plt.ylabel(output_vars[i] if output_vars else f"y{i}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(task_dir, f"{output_vars[i]}_error_r{repeat_id}.pdf"))
            plt.close()
        return

    if x_dim == 2:
        x = inputs[:, 0]
        y = inputs[:, 1]
        for i in range(y_dim):
            _plot_scalar_field(
                x,
                y,
                y_true[:, i],
                title=f"{output_vars[i]} reference",
                out_path=os.path.join(task_dir, f"{output_vars[i]}_ref_r{repeat_id}.pdf"),
                cmap=palette.field_cmap(),
            )
            _plot_scalar_field(
                x,
                y,
                y_pred[:, i],
                title=f"{output_vars[i]} prediction",
                out_path=os.path.join(task_dir, f"{output_vars[i]}_pred_r{repeat_id}.pdf"),
                cmap=palette.field_cmap(),
            )
            _plot_scalar_field(
                x,
                y,
                y_pred[:, i] - y_true[:, i],
                title=f"{output_vars[i]} error",
                out_path=os.path.join(task_dir, f"{output_vars[i]}_err_r{repeat_id}.pdf"),
                cmap=palette.diff_cmap(),
            )

            fig_w, fig_h = ieee_figsize("double", aspect=0.35)
            fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
            for ax, field, title, cmap in [
                (axes[0], y_true[:, i], "reference", palette.field_cmap()),
                (axes[1], y_pred[:, i], "prediction", palette.field_cmap()),
                (axes[2], y_pred[:, i] - y_true[:, i], "error", palette.diff_cmap()),
            ]:
                triang = mtri.Triangulation(x, y)
                tpc = ax.tricontourf(triang, field, levels=32, cmap=cmap)
                fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(title)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            plt.tight_layout()
            plt.savefig(os.path.join(task_dir, f"{output_vars[i]}_triptych_r{repeat_id}.pdf"))
            plt.close(fig)
        return

    print(f"Warning: solution plots skipped for input_dim={x_dim}")


def main() -> int:
    parser = argparse.ArgumentParser(description="PINA benchmark runner (vendored in pinnacle/)")
    parser.add_argument("--name", type=str, default="pina_benchmark")
    parser.add_argument("--device", type=str, default="cpu")  # "cpu" or "0"
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--suite", type=str, default="pina_zoo", choices=["pina_zoo", "ref"])
    parser.add_argument("--cases", type=str, default="Burgers1D,Poisson2D_Classic,HeatMultiscale")
    parser.add_argument("--method", type=str, default=None, choices=[
        "pina_pinn",
        "pina_gradient",
        "pina_causal",
        "pina_competitive",
        "pina_selfadaptive",
        "pina_rba",
        "pina_ensemble",
        "pina_supervised",
    ])
    parser.add_argument("--model", type=str, default="feedforward", choices=["feedforward"])
    parser.add_argument("--hidden", type=str, default="64,64,64")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--eval-n", type=int, default=5000)
    parser.add_argument("--ref-dir", type=str, default="ref")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--plot-column-width", type=str, default="single", choices=["single", "double"])
    parser.add_argument("--ensemble-n", type=int, default=3)
    parser.add_argument(
        "--include-inverse",
        action="store_true",
        help="Include inverse Poisson problem (may require extra deps / may not have closed-form solution).",
    )

    args = parser.parse_args()
    if args.method is None:
        args.method = "pina_supervised" if args.suite == "ref" else "pina_pinn"
    hidden_layers = _parse_hidden_layers(args.hidden)
    if args.suite == "pina_zoo" and args.method == "pina_supervised":
        raise ValueError("method=pina_supervised доступен только для suite=ref.")
    if args.model != "feedforward":
        raise ValueError(f"Unsupported model: {args.model}")

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    exp_name = f"{date_str}-{args.name}-{args.method}"
    os.makedirs(f"runs/{exp_name}", exist_ok=True)

    cases: List[object] = []
    if args.suite == "pina_zoo":
        cases = [
            PinaCase("Poisson2DSquare", Poisson2DSquareProblem),
            PinaCase("Helmholtz", HelmholtzProblem),
            PinaCase("AllenCahn", AllenCahnProblem),
            PinaCase("DiffusionReaction", DiffusionReactionProblem),
            PinaCase("Advection", AdvectionProblem),
            PinaCase("AcousticWave", AcousticWaveProblem),
        ]
        if args.include_inverse:
            # Import lazily: the inverse problem module depends on `requests` and may download data.
            from pina.problem.zoo import InversePoisson2DSquareProblem

            cases.append(
                PinaCase(
                    "InversePoisson2DSquare",
                    # avoid network calls by default
                    lambda: InversePoisson2DSquareProblem(load=False, data_size=0.0),
                )
            )
    else:
        case_names = [c.strip() for c in args.cases.split(",") if c.strip()]
        for name in case_names:
            if name not in REF_CASES:
                raise ValueError(
                    f"Unknown ref case {name!r}. Available: {', '.join(sorted(REF_CASES))}"
                )
            cases.append(REF_CASES[name])

    # Save config (similar to pinnacle/trainer.py)
    with open(f"runs/{exp_name}/config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "repeat": args.repeat,
                "suite": args.suite,
                "cases": [c.name for c in cases],
                "method": args.method,
                "model": args.model,
                "hidden": args.hidden,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "n_train": args.n_train,
                "eval_n": args.eval_n,
                "ensemble_n": args.ensemble_n,
                "ref_dir": args.ref_dir,
                "shuffle": args.shuffle,
            },
            f,
            indent=4,
        )

    accelerator, lightning_device_kwargs = _device_from_arg(args.device)

    # Run cases
    for case_idx, case in enumerate(cases):
        for rep in range(args.repeat):
            run_dir = f"runs/{exp_name}/{case_idx}-{rep}"
            os.makedirs(run_dir, exist_ok=True)

            # log capture (match existing structure)
            log_path = os.path.join(run_dir, "log.txt")
            err_path = os.path.join(run_dir, "logerr.txt")
            stdout_orig = sys.stdout
            stderr_orig = sys.stderr
            sys.stdout = open(log_path, "w", encoding="utf-8")
            sys.stderr = open(err_path, "w", encoding="utf-8")

            try:
                seed = _set_seed(args.seed)
                print(f"***** Begin #{case_idx}-{rep} *****")
                if args.suite == "ref":
                    if args.method != "pina_supervised":
                        raise ValueError(
                            "Для suite=ref поддержан только method=pina_supervised."
                        )
                    print(f"PDE Class Name: PINA_REF_{case.name}")

                    ref_x, ref_y = _load_ref_data(args.ref_dir, case)
                    ref_x, ref_y = _downsample_ref_data(ref_x, ref_y, case.downsample, seed)
                    train_x, train_y, eval_x, eval_y = _split_ref_data(
                        ref_x,
                        ref_y,
                        args.n_train,
                        args.eval_n,
                        seed,
                        args.shuffle,
                    )
                    if eval_x.size == 0:
                        eval_x, eval_y = train_x, train_y

                    train_inputs = LabelTensor(
                        torch.as_tensor(train_x, dtype=torch.float32),
                        labels=case.input_vars,
                    )
                    train_targets = LabelTensor(
                        torch.as_tensor(train_y, dtype=torch.float32),
                        labels=case.output_vars,
                    )
                    problem = SupervisedProblem(
                        train_inputs,
                        train_targets,
                        input_variables=case.input_vars,
                        output_variables=case.output_vars,
                    )
                    model = _build_feedforward(problem, hidden_layers)
                    solver = _build_solver(args.method, problem, model, args.ensemble_n)

                    eval_inputs = LabelTensor(
                        torch.as_tensor(eval_x, dtype=torch.float32),
                        labels=case.input_vars,
                    )
                    eval_targets = torch.as_tensor(eval_y, dtype=torch.float32)
                    eval_metrics_init = _evaluate_ref_case(solver, eval_inputs, eval_targets)
                else:
                    print(f"PDE Class Name: PINA_{case.name}")

                    problem = case.build_problem()
                    # Discretise all condition domains
                    problem.discretise_domain(args.n_train, "random", domains="all")

                    # Build model and solver
                    model = _build_feedforward(problem, hidden_layers)
                    solver = _build_solver(args.method, problem, model, args.ensemble_n)
                    eval_pts = problem.domains["D"].sample(args.eval_n, "random").sort_labels()
                    eval_metrics_init = _evaluate_case_on_points(solver, problem, eval_pts)

                callbacks = [MetricTracker()]
                trainer = Trainer(
                    solver=solver,
                    max_epochs=args.epochs,
                    enable_model_summary=False,
                    callbacks=callbacks,
                    batch_size=args.batch_size,
                    **lightning_device_kwargs,
                )

                t0 = time.time()
                trainer.train()
                t1 = time.time()

                # Extract train loss history from MetricTracker (best-effort)
                train_loss_history = _extract_train_loss_history(callbacks)
                if len(train_loss_history) >= 2:
                    loss_steps = np.linspace(1, args.epochs, len(train_loss_history))
                elif len(train_loss_history) == 1:
                    loss_steps = np.array([0.0, float(args.epochs)])
                    train_loss_history = [train_loss_history[0], train_loss_history[0]]
                else:
                    loss_steps = np.array([0.0, float(args.epochs)])
                    train_loss_history = [float("nan"), float("nan")]

                # Evaluate and write files in pinnacle format
                if args.suite == "ref":
                    eval_metrics_final = _evaluate_ref_case(solver, eval_inputs, eval_targets)
                else:
                    eval_metrics_final = _evaluate_case_on_points(solver, problem, eval_pts)
                _write_loss(os.path.join(run_dir, "loss.txt"), loss_steps, np.array(train_loss_history))
                _write_errors(
                    os.path.join(run_dir, "errors.txt"),
                    np.array([0.0, float(args.epochs)]),
                    [eval_metrics_init, eval_metrics_final],
                )

                # Plot solution/prediction/error if requested
                if args.make_plots:
                    try:
                        if args.suite == "ref":
                            eval_inputs_device = LabelTensor(
                                eval_inputs.to(next(solver.model.parameters()).device),
                                labels=eval_inputs.labels,
                            )
                            y_pred = solver(eval_inputs_device)
                            _plot_solution_outputs(
                                exp_path=f"runs/{exp_name}",
                                task_id=case_idx,
                                repeat_id=rep,
                                input_vars=case.input_vars,
                                output_vars=case.output_vars,
                                inputs=_labeltensor_to_numpy(eval_inputs),
                                y_true=_labeltensor_to_numpy(eval_targets),
                                y_pred=_labeltensor_to_numpy(y_pred),
                                column_width=args.plot_column_width,
                            )
                        else:
                            if hasattr(problem, "solution"):
                                plot_n = max(200, min(args.eval_n, 5000))
                                pts = problem.domains["D"].sample(plot_n, "random").sort_labels()
                                y_true = problem.solution(pts)
                                y_pred = solver(pts)
                                _plot_solution_outputs(
                                    exp_path=f"runs/{exp_name}",
                                    task_id=case_idx,
                                    repeat_id=rep,
                                    input_vars=list(problem.input_variables),
                                    output_vars=list(problem.output_variables),
                                    inputs=_labeltensor_to_numpy(pts),
                                    y_true=_labeltensor_to_numpy(y_true),
                                    y_pred=_labeltensor_to_numpy(y_pred),
                                    column_width=args.plot_column_width,
                                )
                            else:
                                print(f"Warning: no reference solution for {case.name}, skip solution plots.")
                    except Exception as exc:
                        print(f"Warning: solution plot failed for case {case_idx}: {exc}")

                # Make summary parser happy
                print(f"Epoch {args.epochs}: saving model to {run_dir}/{args.epochs}.pt ...")
                # (Optional) save torch state for reproducibility
                try:
                    if hasattr(model, "state_dict"):
                        torch.save(model.state_dict(), os.path.join(run_dir, f"{args.epochs}.pt"))
                except Exception:
                    pass

                print(f"'train' took {t1 - t0:.6f} s")
                print(f"*****  End #{case_idx}-{rep}  *****")

            finally:
                try:
                    sys.stdout.close()
                except Exception:
                    pass
                try:
                    sys.stderr.close()
                except Exception:
                    pass
                sys.stdout = stdout_orig
                sys.stderr = stderr_orig

    # Create summary CSV compatible with existing tool
    try:
        from src.utils import summary as summary_mod

        summary_mod.summary(
            f"runs/{exp_name}",
            tasknum=len(cases),
            repeat=args.repeat,
            iters=[args.epochs] * len(cases),
        )
    except Exception as e:
        print(f"Warning: could not create summary.csv: {e}")

    if args.make_plots:
        _generate_plots(f"runs/{exp_name}", tasknum=len(cases), column_width=args.plot_column_width)

    print("Training completed!")
    print(f"Results saved to: runs/{exp_name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



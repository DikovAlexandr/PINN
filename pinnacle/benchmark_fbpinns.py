import argparse
import json
import os
import random
import sys
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

@dataclass(frozen=True)
class FbpinnsCase:
    name: str  # PDE class name in fbpinns/pdes/*


def _set_seed(seed: Optional[int]) -> int:
    if seed is None:
        seed = random.randint(0, 10**9)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
    return seed


def _device_to_fbpinns(device: str) -> str:
    # fbpinns expects c.DEVICE as int index or "cpu"
    if device == "cpu":
        return "cpu"
    return str(int(device))


def _parse_benchmark_line(line: str) -> Dict[str, float]:
    """
    fbpinns writes a single line (space-separated):
      time train_loss l2re l1re mse mae maxe csve frmse_low frmse_mid frmse_high div_str

    Some metrics may be missing depending on the case; we parse best-effort.
    """
    parts = line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Unexpected benchmark_results line: {line!r}")

    # last token is division string
    div_str = parts[-1]
    nums = parts[:-1]

    def get(i: int) -> float:
        return float(nums[i]) if i < len(nums) else float("nan")

    return {
        "time": get(0),
        "train_loss": get(1),
        "l2re": get(2),
        "l1re": get(3),
        "mse": get(4),
        "mae": get(5),
        "mxe": get(6),
        "csv": get(7),
        "frmse_low": get(8),
        "frmse_mid": get(9),
        "frmse_high": get(10),
        "div": div_str,
    }


def _write_loss(path: str, steps: int, train_loss: float) -> None:
    with open(path, "w", encoding="utf-8") as f:
        # loss.txt is expected to have: step + train/test/weight components.
        # Use a single component with deterministic numeric values so plots work.
        f.write("# step, loss_train, loss_test, loss_weight\n")
        f.write(
            f"{float(0):.18e} "
            f"{float(train_loss):.18e} "
            f"{float(train_loss):.18e} "
            f"{float(1.0):.18e}\n"
        )
        f.write(
            f"{float(steps):.18e} "
            f"{float(train_loss):.18e} "
            f"{float(train_loss):.18e} "
            f"{float(1.0):.18e}\n"
        )


def _write_errors(path: str, steps: int, metrics: Dict[str, float]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# epochs, maes, mses, mxes, l1res, l2res, crmses, frmses(low, mid, high)\n"
        )
        f.write(
            f"{float(0):.18e} "
            f"{metrics.get('mae', float('nan')):.18e} "
            f"{metrics.get('mse', float('nan')):.18e} "
            f"{metrics.get('mxe', float('nan')):.18e} "
            f"{metrics.get('l1re', float('nan')):.18e} "
            f"{metrics.get('l2re', float('nan')):.18e} "
            f"{float('nan'):.18e} "
            f"{metrics.get('frmse_low', float('nan')):.18e} "
            f"{metrics.get('frmse_mid', float('nan')):.18e} "
            f"{metrics.get('frmse_high', float('nan')):.18e}\n"
        )
        f.write(
            f"{float(steps):.18e} "
            f"{metrics.get('mae', float('nan')):.18e} "
            f"{metrics.get('mse', float('nan')):.18e} "
            f"{metrics.get('mxe', float('nan')):.18e} "
            f"{metrics.get('l1re', float('nan')):.18e} "
            f"{metrics.get('l2re', float('nan')):.18e} "
            f"{float('nan'):.18e} "
            f"{metrics.get('frmse_low', float('nan')):.18e} "
            f"{metrics.get('frmse_mid', float('nan')):.18e} "
            f"{metrics.get('frmse_high', float('nan')):.18e}\n"
        )


def _write_loss_history(path: str, steps, train_losses) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# step, loss_train, loss_test, loss_weight\n")
        for step, loss in zip(steps, train_losses):
            f.write(
                f"{float(step):.18e} "
                f"{float(loss):.18e} "
                f"{float(loss):.18e} "
                f"{float(1.0):.18e}\n"
            )


def _write_errors_history(path: str, steps, l2re_hist, mse_hist) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# epochs, maes, mses, mxes, l1res, l2res, crmses, frmses(low, mid, high)\n")
        for step, l2re, mse in zip(steps, l2re_hist, mse_hist):
            f.write(
                f"{float(step):.18e} "
                f"{float('nan'):.18e} "
                f"{float(mse):.18e} "
                f"{float('nan'):.18e} "
                f"{float('nan'):.18e} "
                f"{float(l2re):.18e} "
                f"{float('nan'):.18e} "
                f"{float('nan'):.18e} {float('nan'):.18e} {float('nan'):.18e}\n"
            )


def _load_latest_loss_history(model_out_dir: str):
    if not os.path.isdir(model_out_dir):
        return None
    candidates = [
        os.path.join(model_out_dir, name)
        for name in os.listdir(model_out_dir)
        if name.startswith("loss_") and name.endswith(".npy")
    ]
    if not candidates:
        return None
    latest = sorted(candidates)[-1]
    try:
        import numpy as np

        data = np.load(latest)
        return data
    except Exception:
        return None


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
            histories = {"FBPINNs": results["loss"][task_key]}
            try:
                plot_statistical_histories_fbpinns(
                    histories,
                    metric="train",
                    output_path=os.path.join(task_dir, "train_loss.pdf"),
                )
            except Exception as exc:
                print(f"Warning: plot train loss failed for {task_key}: {exc}")

        if task_key in results.get("metrics", {}):
            histories = {"FBPINNs": results["metrics"][task_key]}
            for metric in ("l2re", "mse"):
                try:
                    plot_statistical_histories_fbpinns(
                        histories,
                        metric=metric,
                        output_path=os.path.join(task_dir, f"{metric}.pdf"),
                    )
                except Exception as exc:
                    print(f"Warning: plot {metric} failed for {task_key}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="FBPINNs benchmark runner (vendored in pinnacle/fbpinns)")
    parser.add_argument("--name", type=str, default="fbpinns_benchmark")
    parser.add_argument("--device", type=str, default="cpu")  # "cpu" or "0"
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--net", type=str, default="fbpinn", choices=["fbpinn", "pinn"])
    parser.add_argument("--cases", type=str, default="Burgers1D,Poisson2D_Classic,HeatMultiscale")
    parser.add_argument("--n-steps", type=int, default=20000)
    parser.add_argument("--plot-freq", type=int, default=0)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--plot-column-width", type=str, default="single", choices=["single", "double"])

    args = parser.parse_args()

    date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
    exp_name = f"{date_str}-{args.name}-fbpinns-{args.net}"
    runs_root = os.path.abspath(f"runs/{exp_name}")
    os.makedirs(runs_root, exist_ok=True)

    cases = [FbpinnsCase(name=c.strip()) for c in args.cases.split(",") if c.strip()]

    with open(os.path.join(runs_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "repeat": args.repeat,
                "net": args.net,
                "cases": [c.name for c in cases],
                "n_steps": args.n_steps,
                "device": args.device,
            },
            f,
            indent=4,
        )

    # Prepare import path so `import problems`, etc. resolve inside fbpinns
    base_dir = os.path.dirname(__file__)
    fbp_dir = os.path.join(base_dir, "fbpinns")

    # We'll execute fbpinns training via importing its modules, but with cwd set to fbpinns/
    cwd0 = os.getcwd()
    sys_path0 = list(sys.path)

    try:
        os.chdir(fbp_dir)
        sys.path.insert(0, ".")
        sys.path.insert(0, "pdes")

        import numpy as np

        from models import models
        from config.constants import Constants, get_subdomain_ws
        from main import FBPINNTrainer
        from training.active_schedulers import AllActiveSchedulerND

        # load per-case configs
        conf_all = json.load(open("runs/run_all_config.json", "r"))
        conf_flat = {}
        for _, v in conf_all.items():
            for kk, vv in v.items():
                conf_flat[kk] = vv

        pde_mods = {}
        # Import all pde classes used by upstream run_all.py
        from ns import NS_Long, NS_NoObstacle, LidDrivenFlow  # noqa: F401
        from heat import HeatMultiscale, HeatComplex, HeatLongTime, HeatDarcy, HeatND  # noqa: F401
        from poisson import Poisson2D_Classic, Poisson2D_Hole, PoissonND, Poisson3D, Poisson2DManyArea  # noqa: F401
        from wave import WaveEquation1D, Wave2DLong, WaveHetergeneous  # noqa: F401
        from chaotic import Kuramoto, GrayScott  # noqa: F401
        from burger import Burgers1D  # noqa: F401
        from inverse import PoissonInv, HeatInv  # noqa: F401

        for case_idx, case in enumerate(cases):
            if case.name not in conf_flat:
                raise ValueError(f"Unknown case {case.name!r} (not in fbpinns/runs/run_all_config.json)")

            for rep in range(args.repeat):
                run_dir = os.path.join(runs_root, f"{case_idx}-{rep}")
                os.makedirs(run_dir, exist_ok=True)

                # capture logs like other benchmark runners
                log_path = os.path.join(run_dir, "log.txt")
                err_path = os.path.join(run_dir, "logerr.txt")
                stdout_orig = sys.stdout
                stderr_orig = sys.stderr
                sys.stdout = open(log_path, "w", encoding="utf-8")
                sys.stderr = open(err_path, "w", encoding="utf-8")

                try:
                    seed = _set_seed(args.seed)
                    print(f"***** Begin #{case_idx}-{rep} *****")
                    print(f"PDE Class Name: FBPINNS_{case.name}")
                    print(f"Seed: {seed}")

                    # instantiate problem class
                    case_classes = {
                        "NS_Long": NS_Long,
                        "NS_NoObstacle": NS_NoObstacle,
                        "LidDrivenFlow": LidDrivenFlow,
                        "HeatMultiscale": HeatMultiscale,
                        "HeatComplex": HeatComplex,
                        "HeatLongTime": HeatLongTime,
                        "HeatDarcy": HeatDarcy,
                        "HeatND": HeatND,
                        "Poisson2D_Classic": Poisson2D_Classic,
                        "Poisson2D_Hole": Poisson2D_Hole,
                        "PoissonND": PoissonND,
                        "Poisson3D": Poisson3D,
                        "Poisson2DManyArea": Poisson2DManyArea,
                        "WaveEquation1D": WaveEquation1D,
                        "Wave2DLong": Wave2DLong,
                        "WaveHetergeneous": WaveHetergeneous,
                        "Kuramoto": Kuramoto,
                        "GrayScott": GrayScott,
                        "Burgers1D": Burgers1D,
                        "PoissonInv": PoissonInv,
                        "HeatInv": HeatInv,
                    }
                    if case.name not in case_classes:
                        raise RuntimeError(f"Case class not available in wrapper mapping: {case.name}")
                    P = case_classes[case.name]()

                    conf_this = conf_flat[case.name]
                    boundary_batch_size = int(np.prod(np.array(conf_this["ba"])) // 4)
                    boundary_batch_size_test = int(np.prod(np.array(conf_this["ba_t"])) // 4)

                    boundary_weight = 100
                    data_weight = 1

                    if args.net == "fbpinn":
                        subdomain_xs = [
                            np.linspace(l, r, seg + 1)
                            for l, r, seg in zip(P.bbox[::2], P.bbox[1::2], conf_this["div"])
                        ]
                    else:
                        subdomain_xs = [np.array([l, r]) for l, r in zip(P.bbox[::2], P.bbox[1::2])]

                    width = 0.6
                    subdomain_ws = get_subdomain_ws(subdomain_xs, width)

                    # crude heuristic from upstream script
                    n_models = 1
                    for seg in conf_this["div"]:
                        n_models *= seg
                    n_hidden, n_layers = (64, 4) if n_models <= 5 else (16, 2)
                    usemodel = models.BiFCN if case.name.endswith("Inv") else models.FCN

                    grid = "x".join([str(len(sx) - 1) for sx in subdomain_xs])
                    bdw = ("_bdw" + str(boundary_weight) if hasattr(P, "sample_bd") else "") + (
                        "_dw" + str(data_weight) if hasattr(P, "sample_data") else ""
                    )

                    # Redirect fbpinns internal outputs under this run_dir (no more fbpinns/benchmark_results pollution)
                    os.environ["FBPINNS_RESULTS_ROOT"] = os.path.join(run_dir, "fbpinns_results") + os.sep
                    os.environ["FBPINNS_BENCHMARK_RESULTS_ROOT"] = os.path.join(run_dir, "fbpinns_benchmark_results") + os.sep

                    c = Constants(
                        RUN=f"pinnacle_{grid}{bdw}_{P.name}_{n_hidden}h_{n_layers}l_{conf_this['ba'][0]}b_{width}w",
                        P=P,
                        SUBDOMAIN_XS=subdomain_xs,
                        SUBDOMAIN_WS=subdomain_ws,
                        BOUNDARY_N=(1,),
                        Y_N=(0, 1),
                        ACTIVE_SCHEDULER=AllActiveSchedulerND,
                        ACTIVE_SCHEDULER_ARGS=(),
                        MODEL=usemodel,
                        N_HIDDEN=n_hidden,
                        N_LAYERS=n_layers,
                        BATCH_SIZE=tuple(conf_this["ba"]),
                        BOUNDARY_BATCH_SIZE=boundary_batch_size,
                        BOUNDARY_WEIGHT=boundary_weight,
                        DATALOSS_WEIGHT=data_weight,
                        RANDOM=True if case.name.endswith("Inv") else False,
                        N_STEPS=args.n_steps,
                        BATCH_SIZE_TEST=tuple(conf_this["ba_t"]),
                        BOUNDARY_BATCH_SIZE_TEST=boundary_batch_size_test,
                        PLOT_LIMS=(1.1, False),
                        # Loss history + plots are only produced during `_test_step`.
                        # `_test_step` is executed each `SUMMARY_FREQ` steps, and plots are
                        # generated when `TEST_FREQ` also hits. So keep them equal.
                        TEST_FREQ=max(
                            1,
                            int(args.plot_freq)
                            if int(args.plot_freq) > 0
                            else max(1, int(math.ceil(args.n_steps / 20))),
                        ),
                        SUMMARY_FREQ=max(
                            1,
                            int(args.plot_freq)
                            if int(args.plot_freq) > 0
                            else max(1, int(math.ceil(args.n_steps / 20))),
                        ),
                        DEVICE=_device_to_fbpinns(args.device),
                        SEED=seed,
                    )
                    # Ensure loss history is persisted frequently enough for plots.
                    c.MODEL_SAVE_FREQ = c.SUMMARY_FREQ

                    t0 = time.time()
                    FBPINNTrainer(c).train()
                    t1 = time.time()

                    # Parse the fbpinns single-line benchmark output and translate to pinnacle format
                    bench_root = os.environ["FBPINNS_BENCHMARK_RESULTS_ROOT"]
                    mode_dir = "fb" if args.net == "fbpinn" and n_models > 1 else "ctrl"
                    bench_file = os.path.join(bench_root, mode_dir, f"{c.P.name}_{c.SEED}")
                    with open(bench_file, "r") as f:
                        line = f.read()
                    parsed = _parse_benchmark_line(line)

                    # Write pinnacle-compatible artifacts
                    loss_hist = _load_latest_loss_history(c.MODEL_OUT_DIR)
                    if loss_hist is not None and loss_hist.size > 0:
                        steps = loss_hist[:, 0]
                        train_loss_hist = loss_hist[:, -1]
                        n_yj = c.P.d[1]
                        l2re_hist = loss_hist[:, 3]
                        mse_idx = 3 + n_yj
                        mse_hist = loss_hist[:, mse_idx] if loss_hist.shape[1] > mse_idx else np.full_like(steps, np.nan)
                        _write_loss_history(os.path.join(run_dir, "loss.txt"), steps, train_loss_hist)
                        _write_errors_history(os.path.join(run_dir, "errors.txt"), steps, l2re_hist, mse_hist)
                    else:
                        _write_loss(os.path.join(run_dir, "loss.txt"), args.n_steps, parsed["train_loss"])
                        _write_errors(
                            os.path.join(run_dir, "errors.txt"),
                            args.n_steps,
                            {
                                "mae": parsed["mae"],
                                "mse": parsed["mse"],
                                "mxe": parsed["mxe"],
                                "l1re": parsed["l1re"],
                                "l2re": parsed["l2re"],
                                "frmse_low": parsed["frmse_low"],
                                "frmse_mid": parsed["frmse_mid"],
                                "frmse_high": parsed["frmse_high"],
                            },
                        )

                    # Make summary parser happy
                    print(f"Epoch {args.n_steps}: saving model to {run_dir}/{args.n_steps}.pt ...")
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

    finally:
        os.chdir(cwd0)
        sys.path = sys_path0

    # Create summary CSV compatible with existing tool
    try:
        from src.utils import summary as summary_mod

        summary_mod.summary(
            f"runs/{exp_name}",
            tasknum=len(cases),
            repeat=args.repeat,
            iters=[args.n_steps] * len(cases),
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



__all__ = ["set_LBFGS_options", "set_Muon_options"]

from ..backend import backend_name


LBFGS_options = {}
Muon_options = {}


def set_LBFGS_options(
    maxcor=100,
    ftol=0,
    gtol=1e-8,
    maxiter=15000,
    maxfun=None,
    maxls=50,
):
    """Sets the hyperparameters of L-BFGS.

    The L-BFGS optimizer used in each backend:

    - TensorFlow 1.x: `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb>`_
    - TensorFlow 2.x: `tfp.optimizer.lbfgs_minimize <https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize>`_
    - PyTorch: `torch.optim.LBFGS <https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html>`_
    - Paddle: `paddle.incubate.optimizers.LBFGS <https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/incubate/optimizer/LBFGS_en.html>`_

    I find empirically that torch.optim.LBFGS and scipy.optimize.minimize are better than
    tfp.optimizer.lbfgs_minimize in terms of the final loss value.

    Args:
        maxcor (int): `maxcor` (scipy), `num_correction_pairs` (tfp), `history_size` (torch), `history_size` (paddle).
            The maximum number of variable metric corrections used to define the limited
            memory matrix. (The limited memory BFGS method does not store the full
            hessian but uses this many terms in an approximation to it.)
        ftol (float): `ftol` (scipy), `f_relative_tolerance` (tfp), `tolerance_change` (torch), `tolerance_change` (paddle).
            The iteration stops when `(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol`.
        gtol (float): `gtol` (scipy), `tolerance` (tfp), `tolerance_grad` (torch), `tolerance_grad` (paddle).
            The iteration will stop when `max{|proj g_i | i = 1, ..., n} <= gtol` where
            `pg_i` is the i-th component of the projected gradient.
        maxiter (int): `maxiter` (scipy), `max_iterations` (tfp), `max_iter` (torch), `max_iter` (paddle).
            Maximum number of iterations.
        maxfun (int): `maxfun` (scipy), `max_eval` (torch), `max_eval` (paddle).
            Maximum number of function evaluations. If ``None``, `maxiter` * 1.25.
        maxls (int): `maxls` (scipy), `max_line_search_iterations` (tfp).
            Maximum number of line search steps (per iteration).

    Warning:
        If L-BFGS stops earlier than expected, set the default float type to 'float64':

        .. code-block:: python

            dde.config.set_default_float("float64")
    """
    global LBFGS_options
    LBFGS_options["maxcor"] = maxcor
    LBFGS_options["ftol"] = ftol
    LBFGS_options["gtol"] = gtol
    LBFGS_options["maxiter"] = maxiter
    LBFGS_options["maxfun"] = maxfun if maxfun is not None else int(maxiter * 1.25)
    LBFGS_options["maxls"] = maxls


set_LBFGS_options()


def set_Muon_options(
    momentum=0.95,
    nesterov=True,
    ns_coefficients=(3.4445, -4.775, 2.0315),
    eps=1e-07,
    ns_steps=5,
):
    """Sets the hyperparameters of Muon optimizer.

    Muon is a momentum-based optimizer with Newton-Schulz preconditioner,
    designed for efficient training of neural networks.

    Reference:
        PyTorch Muon optimizer: `torch.optim.Muon <https://pytorch.org/docs/stable/generated/torch.optim.Muon.html>`_

    Args:
        momentum (float): Momentum factor (default: 0.95).
            Controls the exponential moving average of gradients.
        nesterov (bool): Whether to use Nesterov momentum (default: True).
            Nesterov momentum provides better convergence in many cases.
        ns_coefficients (tuple): Coefficients for Newton-Schulz iteration (default: (3.4445, -4.775, 2.0315)).
            These coefficients control the Newton-Schulz preconditioner computation.
        eps (float): Small value for numerical stability (default: 1e-07).
            Prevents division by zero in computations.
        ns_steps (int): Number of Newton-Schulz iteration steps (default: 5).
            More steps can improve convergence but increase computational cost.

    Example:
        .. code-block:: python

            import deepxde as dde
            
            # Use default Muon settings
            model.compile("muon", lr=0.001)
            
            # Or customize Muon settings
            dde.optimizers.set_Muon_options(momentum=0.9, ns_steps=3)
            model.compile("muon", lr=0.001)

    Note:
        - Muon is particularly effective for large-scale neural network training
        - The learning rate (lr) and weight_decay are set during model.compile()
        - Higher ns_steps may improve convergence but slow down training
        - **IMPORTANT**: Muon only works with 2D parameters (weight matrices).
          For neural networks with bias terms (1D parameters), the system will
          automatically apply Muon to weights and Adam to biases in a hybrid approach.
          For best performance with Muon, consider using networks without bias terms.
    """
    global Muon_options
    Muon_options["momentum"] = momentum
    Muon_options["nesterov"] = nesterov
    Muon_options["ns_coefficients"] = ns_coefficients
    Muon_options["eps"] = eps
    Muon_options["ns_steps"] = ns_steps


set_Muon_options()


# Backend-dependent options
if backend_name in ["pytorch", "paddle"]:
    # number of iterations per optimization call
    LBFGS_options["iter_per_step"] = min(1000, LBFGS_options["maxiter"])
    LBFGS_options["fun_per_step"] = (
        LBFGS_options["maxfun"]
        * LBFGS_options["iter_per_step"]
        // LBFGS_options["maxiter"]
    )

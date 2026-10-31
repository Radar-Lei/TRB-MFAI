"""Four attribution methods (M1-M4) under the frozen training protocol.

Frozen (contract + appendix decisions):
  * Backbone: MLP 3x128 tanh, float64, Adam lr 3e-3, full-batch, stop when
    grad-norm < 1e-6 or 20000 steps; seeds s in {0..4}.
  * M1 soft value penalty:   loss = MSE(f, F) + lam * mean ||B f - M q||^2,
    lam in {0, 1, 100, 1e4}.
  * M2 Jacobian penalty:     loss = MSE + mu * mean_{stencil} ||J_r||_F^2,
    J_r = B J_f - M; stencil = 16-sample subbatch x 4 random unit directions
    (paired +/-, 128 stencil points per step, cost-tuned for Sioux Falls) x
    rho drawn from {0.05, 0.1} each step; mu = 100 (aligned with
    M1@lam=100; mu in {1, 100} sensitivity recorded in E2 appendix).
  * M3 hard non-negative path simplex: f = Delta h, h_j = q_j * softmax_j(z(q));
    trained with the same link-flow MSE (no privileged path-flow supervision).
  * M4 post-hoc projection (no training): acts on M1@lam=100 flow fields,
    W in {I, diag(c_e)}, column-wise T4 projection onto B a = M_j Delta_q_j.

Error matching (frozen): eps* = max over methods of the minimal val error;
methods whose val error can be brought into [0.95 eps*, 1.05 eps*] use the
first checkpoint entering that band (checkpoints saved every EVAL_EVERY steps
in memory); unreachable methods exit the comparison and enter the Pareto plot
(val error x P1) as secondary evidence.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap

from sue import PathStructure

TORCH_DTYPE = torch.float64
HIDDEN = 128
DEPTH = 3
LR = 3e-3
# Frozen protocol says 20000; the shared machine's effective CPU share makes
# that infeasible in available wall-clock (measured ~10-25% CPU share).
# Downgraded to 6000 with the error-matching band as the built-in guard: any
# run that has not entered [0.95 eps*, 1.05 eps*] by step 6000 is flagged
# out-of-band and takes the frozen Pareto fallback path -- never a silent
# low-quality match.  Recorded in EXPERIMENT_RESULTS.md changelog.
MAX_STEPS = 6000
GRAD_TOL = 1e-6
EVAL_EVERY = 250
M2_MU = 100.0
M2_STENCIL_BATCH = 16       # 16 samples x 8 points = 128 stencil points/step
                            # (cost-tuned for Sioux Falls; kept for all networks)
M2_DIRS = 4
M2_RHO_GRID = (0.05, 0.1)
LAMBDA_GRID = (0.0, 1.0, 100.0, 1e4)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class SoftMLP(nn.Module):
    """M1 / M2: q (K) -> f (m), unconstrained link flows."""

    def __init__(self, K: int, m: int):
        super().__init__()
        layers = []
        d_in = K
        for _ in range(DEPTH):
            layers += [nn.Linear(d_in, HIDDEN), nn.Tanh()]
            d_in = HIDDEN
        layers.append(nn.Linear(d_in, m))
        self.net = nn.Sequential(*layers)
        self.double()

    def forward(self, q):
        return self.net(q)


class PathSimplexMLP(nn.Module):
    """M3: q -> z (P logits) -> h (path flows, per-OD simplex scaled by q_j)
    -> f = Delta h.  Guarantees f(0)=0 and B f = M q exactly."""

    def __init__(self, ps: PathStructure):
        super().__init__()
        self.register_buffer("Delta", torch.tensor(ps.Delta, dtype=TORCH_DTYPE))
        self.register_buffer("q_zero", torch.zeros(1))
        self.od_path_idx = [torch.tensor(ix, dtype=torch.long) for ix in ps.od_path_idx]
        K, P, m = ps.K, ps.P, ps.m
        layers = []
        d_in = K
        for _ in range(DEPTH):
            layers += [nn.Linear(d_in, HIDDEN), nn.Tanh()]
            d_in = HIDDEN
        layers.append(nn.Linear(d_in, P))
        self.net = nn.Sequential(*layers)
        self.double()

    def forward(self, q):                       # q (batch, K)
        z = self.net(q)                          # (batch, P)
        h = torch.zeros_like(z)
        for j, idx in enumerate(self.od_path_idx):
            zz = z[:, idx]
            zz = zz - zz.max(dim=1, keepdim=True).values
            w = torch.exp(zz)
            h[:, idx] = q[:, j:j + 1] * w / w.sum(dim=1, keepdim=True)
        return h @ self.Delta.t()                # (batch, m)


def build_model(method: str, ps: PathStructure, seed: int, norm: dict = None) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if norm is not None:
        # E4 scale-controlled regime: same backbone, normalized input/output
        # interfaces (norm dict carries mu_q/sd_q/mu_f/sd_f/s_node tensors).
        if method in ("M1", "M2"):
            return NormSoftMLP(ps.K, ps.m, norm)
        if method == "M3":
            return NormPathSimplexMLP(ps, norm)
        raise ValueError(method)
    if method in ("M1", "M2"):
        return SoftMLP(ps.K, ps.m)
    if method == "M3":
        return PathSimplexMLP(ps)
    raise ValueError(method)


class NormSoftMLP(SoftMLP):
    """E4: q (K) -> f (m) with standardized input/output interfaces.
    forward(q_raw) = sd_f * net((q_raw - mu_q)/sd_q) + mu_f  -- the module still
    consumes and produces RAW units, so every downstream metric (P1-P4, val
    error) is unchanged; only the internal conditioning is normalized."""

    def __init__(self, K: int, m: int, norm: dict):
        super().__init__(K, m)
        for k in ("mu_q", "sd_q", "mu_f", "sd_f"):
            self.register_buffer(k, torch.tensor(np.asarray(norm[k], dtype=float),
                                                 dtype=TORCH_DTYPE))

    def forward(self, q):
        x = (q - self.mu_q) / self.sd_q
        return self.net(x) * self.sd_f + self.mu_f


class NormPathSimplexMLP(PathSimplexMLP):
    """E4: M3 with a standardized input interface for the logits network.
    The per-OD simplex multiplier keeps the RAW demand q_j (intrinsic to the
    hard constraint Bf = Mq -- normalizing it away would break the identity),
    so the structural channel is preserved while z(q) is well-conditioned."""

    def __init__(self, ps: PathStructure, norm: dict):
        super().__init__(ps)
        for k in ("mu_q", "sd_q"):
            self.register_buffer(k, torch.tensor(np.asarray(norm[k], dtype=float),
                                                 dtype=TORCH_DTYPE))

    def forward(self, q):
        z = self.net((q - self.mu_q) / self.sd_q)
        h = torch.zeros_like(z)
        for j, idx in enumerate(self.od_path_idx):
            zz = z[:, idx]
            zz = zz - zz.max(dim=1, keepdim=True).values
            w = torch.exp(zz)
            h[:, idx] = q[:, j:j + 1] * w / w.sum(dim=1, keepdim=True)
        return h @ self.Delta.t()


class CycleAugmentedField(nn.Module):
    """Exactly conserving analytic field f = Delta h + N z(q) used by the E1
    theorem-verification package: the path-simplex part carries OD structure,
    and the cycle component N z(q) (N = orthonormal basis of ker B, B N = 0)
    excites atoms that no OD path crosses (pendant loops).  B f = M q to
    machine precision for every q."""

    def __init__(self, ps: PathStructure, seed: int):
        super().__init__()
        torch.manual_seed(seed)                 # seed BEFORE constructing inner
        self.inner = PathSimplexMLP(ps)
        N = ps.net.nullspace_basis()                    # (m, mu)
        self.register_buffer("N", torch.tensor(N, dtype=TORCH_DTYPE))
        mu = N.shape[1]
        torch.manual_seed(seed + 1)
        self.znet = nn.Sequential(nn.Linear(ps.K, 32), nn.Tanh(),
                                  nn.Linear(32, mu)).double()

    def forward(self, q):
        return self.inner(q) + self.znet(q) @ self.N.t()


# ----------------------------------------------------------------------------
# v17: frozen analytic-field checkpoints
# ----------------------------------------------------------------------------
FROZEN_FIELDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "frozen_fields")


def load_or_freeze_field(ps: PathStructure, seed: int, tag: str | None = None,
                         strict_replay: bool | None = None):
    """Build the analytic CycleAugmentedField once, persist its FULL
    state_dict (the N buffer and every weight), and reload that exact tensor
    set on every later construction.

    Why: a fixed seed fixes the weight draws but not the SVD nullspace basis
    N -- across LAPACK/numpy versions N may change by column signs (removed
    by nets.Network.nullspace_basis's canonical orientation) or, for
    degenerate singular values, by an in-subspace rotation, and N'z != Nz
    then realizes a DIFFERENT analytic field.  Freezing the state_dict
    freezes the field itself; the frozen .pt files ship with the
    reproducibility package.

    strict_replay distinguishes two modes (external review, v18):
      * strict replay (True): the frozen checkpoint MUST exist -- a missing
        file raises instead of silently constructing a different new field.
        Set for the packaged run_all.sh path, where every field is shipped.
      * construct new frozen field (False): the historical development
        behavior -- a missing checkpoint is created on first use.
      * None (default): read the environment: STRICT_REPLAY=1 selects
        strict mode.  run_all.sh exports STRICT_REPLAY=1.
    """
    import os as _os
    if strict_replay is None:
        strict_replay = _os.environ.get("STRICT_REPLAY", "0") == "1"
    name = tag or f"{ps.net.name}_K{ps.K}_seed{seed}.pt"
    path = _os.path.join(FROZEN_FIELDS_DIR, name)
    if _os.path.exists(path):
        model = CycleAugmentedField(ps, seed)          # same shapes/dtypes
        sd = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        return model
    if strict_replay:
        raise FileNotFoundError(
            f"strict replay: frozen field checkpoint missing: {path}. "
            "The reproducibility package ships every checkpoint under "
            "experiments/frozen_fields/; a strict re-run must load the "
            "packaged field, not construct a new one. To construct a new "
            "frozen field deliberately, unset STRICT_REPLAY or pass "
            "strict_replay=False.")
    _os.makedirs(FROZEN_FIELDS_DIR, exist_ok=True)
    model = CycleAugmentedField(ps, seed)
    torch.save(model.state_dict(), path)
    return model


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def _val_err(model, qv, Fv):
    with torch.no_grad():
        f = model(qv)
        return float(((f - Fv).norm(dim=1) / Fv.norm(dim=1).clamp_min(1e-12)).mean())


def _m2_stencil_penalty(model, qt, Bt, Mt, rng: np.random.Generator,
                        s_node=None, sd_q=None):
    """mean ||J_r||_F^2 over stencil points around a subbatch of training
    demands.  Returns a TENSOR so the penalty stays in the autograd graph
    (returning float() here silently disabled M2 entirely).
    E4 dimensionless form: with s_node/sd_q given, the penalized matrix is
    S^-1 (B J_f - M) diag(sd_q)  (S = diag(s_node)) -- flow/demand units cancel
    and mu becomes a scale-free coefficient comparable across regimes."""
    from attribution import forward_jacobian
    n = qt.shape[0]
    sel = rng.integers(0, n, size=M2_STENCIL_BATCH)
    q_sub = qt[sel]
    K = q_sub.shape[1]
    dirs = rng.normal(size=(M2_DIRS, K))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rho = float(M2_RHO_GRID[rng.integers(0, len(M2_RHO_GRID))])
    pts = []
    for d in dirs:
        pts.append(q_sub + rho * torch.tensor(d, dtype=TORCH_DTYPE))
        pts.append(q_sub - rho * torch.tensor(d, dtype=TORCH_DTYPE))
    X = torch.cat(pts, dim=0)                            # (batch*2*dirs, K)
    J_f = forward_jacobian(model, X)                     # (n_pts, m, K)
    # J_r[p] = B J_f[p] - M  ->  (n_pts, n, K)
    J_r = torch.einsum("ij,pjk->pik", Bt, J_f) - Mt.unsqueeze(0)
    if s_node is not None and sd_q is not None:
        J_r = (J_r / s_node.unsqueeze(-1)) @ torch.diag(sd_q)
    return (J_r ** 2).sum(dim=(1, 2)).mean()


def train_model(method: str, ps: PathStructure, data: dict, split: dict,
                seed: int, lam: float = None, mu: float = None,
                keep_checkpoints: bool = True, quiet: bool = True,
                norm: dict = None, max_steps: int = None, eval_every: int = None,
                plateau_patience: int = None, loss_std: bool = None):
    """Full-batch training with val curve + periodic checkpoints (in memory).

    data: {'q': (N,K), 'F': (N,m)};  split: {'train': idx, 'val': idx}
    Returns dict with val_curve, checkpoints (state_dicts), best_val_err.

    E4 extensions (all default-off, preserving the frozen E2/E3 behavior):
      * norm: normalization-stats dict -> standardized model interfaces, MSE in
        normalized f-hat units, and DIMENSIONLESS penalties:
          M1: lam_tilde * mean_n ((B f - M q)_n / s_node_n)^2
          M2: mu_tilde * mean over stencil of ||S^-1 (B J_f - M) diag(sd_q)||_F^2
        (S = diag(s_node)); lam/mu passed in the grid are the dimensionless
        tilde coefficients.
      * loss_std (E5 factor decoupling): if True, the FIT loss is the per-edge
        standardized MSE ((f-F)/sd_f)^2 even when norm is None (raw inputs);
        if False with norm given, raw-unit MSE with normalized inputs.
        Default None: std loss iff norm is not None (the E4 coupling).
      * max_steps / eval_every: per-experiment overrides.
      * plateau_patience: stop when the val curve has not improved by >1% for
        this many consecutive evals (convergence check requested by review)."""
    Bt = torch.tensor(ps.net.B, dtype=TORCH_DTYPE)
    Mt = torch.tensor(ps.net.od_M(ps.ods), dtype=TORCH_DTYPE)
    qt = torch.tensor(data["q"][split["train"]], dtype=TORCH_DTYPE)
    Ft = torch.tensor(data["F"][split["train"]], dtype=TORCH_DTYPE)
    qv = torch.tensor(data["q"][split["val"]], dtype=TORCH_DTYPE)
    Fv = torch.tensor(data["F"][split["val"]], dtype=TORCH_DTYPE)
    n_steps = max_steps or MAX_STEPS
    every = eval_every or EVAL_EVERY
    if loss_std is None:
        loss_std = norm is not None

    sd_f_t = None
    if loss_std:
        # Round-3 fix (ruling W2.2): the FIT-LOSS weights always come from the
        # training data, decoupled from the model's output-interface sd_f
        # (E5b sets the interface sd_f to 1 for its input-only cells; before
        # this fix that silently made the std loss identical to the raw loss).
        # For E4/E5 runs norm["sd_f"] == this value, so behavior is unchanged.
        sd_f_t = torch.tensor(
            np.asarray(data["F"][split["train"]].std(axis=0) + 1e-9, dtype=float),
            dtype=TORCH_DTYPE)
    if norm is not None:
        model = build_model(method, ps, seed, norm=norm)
        sd_q = torch.tensor(np.asarray(norm["sd_q"], dtype=float), dtype=TORCH_DTYPE)
        s_node = torch.tensor(np.asarray(norm["s_node"], dtype=float), dtype=TORCH_DTYPE)
    else:
        model = build_model(method, ps, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed + 777)             # stencil sampling stream
    use_value_pen = method == "M1" and lam is not None and lam > 0
    use_jac_pen = method == "M2"

    val_curve, steps_log, checkpoints = [], [], []
    stopped_by = "max_steps"
    since_improve = 0
    best_so_far = np.inf
    for step in range(1, n_steps + 1):
        opt.zero_grad()
        f = model(qt)
        if sd_f_t is not None:
            loss = (((f - Ft) / sd_f_t) ** 2).mean()
        else:
            loss = ((f - Ft) ** 2).mean()
        if use_value_pen:
            cons = (torch.einsum("ij,jk->ik", Bt, f.t()).t() - qt @ Mt.t())
            if norm is not None:
                loss = loss + lam * ((cons / s_node) ** 2).sum(dim=1).mean()
            else:
                loss = loss + lam * (cons ** 2).sum(dim=1).mean()
        if use_jac_pen:
            if norm is not None:
                loss = loss + (mu or M2_MU) * _m2_stencil_penalty(
                    model, qt, Bt, Mt, rng, s_node=s_node, sd_q=sd_q)
            else:
                loss = loss + (mu or M2_MU) * _m2_stencil_penalty(model, qt, Bt, Mt, rng)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e10))
        opt.step()
        if step % every == 0:
            ve = _val_err(model, qv, Fv)
            val_curve.append(ve)
            steps_log.append(step)
            if keep_checkpoints:
                checkpoints.append(copy.deepcopy(model.state_dict()))
            if not quiet and step % 2000 == 0:
                print(f"    {method} seed={seed} step={step} val={ve:.4e} gnorm={gnorm:.2e}")
            if plateau_patience is not None:
                if ve < best_so_far * 0.99:
                    best_so_far = ve
                    since_improve = 0
                else:
                    since_improve += 1
                    if since_improve >= plateau_patience:
                        stopped_by = "plateau"
                        break
        if gnorm < GRAD_TOL:
            stopped_by = "grad_tol"
            break

    best_i = int(np.argmin(val_curve)) if val_curve else 0
    return {
        "method": method, "seed": seed, "lam": lam, "mu": mu,
        "val_curve": val_curve, "steps_log": steps_log,
        "checkpoints": checkpoints, "best_val_err": val_curve[best_i] if val_curve else None,
        "best_step": steps_log[best_i] if steps_log else 0,
        "final_state": copy.deepcopy(model.state_dict()),
        "stopped_by": stopped_by,
    }


def select_matched_checkpoint(train_out: dict, eps_star: float, band=0.05,
                              two_sided: bool = False):
    """Error matching checkpoint selection.

    Default (E2/E3 frozen semantics): FIRST checkpoint whose val error drops
    to eps*(1+band) or below -- one-sided early stop at reached accuracy.

    two_sided=True (E4): FIRST checkpoint inside [eps*(1-band), eps*(1+band)]
    -- the strict preregistered band, reachable because E4 samples the val
    curve densely (every 50 steps).  Runs that never enter the band take their
    best checkpoint and are flagged out-of-band (Pareto fallback).
    Returns (state, in_band, step, val_err_of_selected_checkpoint)."""
    lo = eps_star * (1 - band)
    hi = eps_star * (1 + band)
    for i, ve in enumerate(train_out["val_curve"]):
        if (lo <= ve <= hi) if two_sided else (ve <= hi):
            return (train_out["checkpoints"][i], True, train_out["steps_log"][i],
                    float(ve))
    bi = int(np.argmin(train_out["val_curve"]))
    return (train_out["checkpoints"][bi], False, train_out["steps_log"][bi],
            float(train_out["val_curve"][bi]))


def eps_star_from(train_outs: list[dict]) -> float:
    """eps* = max over methods of min val error (frozen)."""
    return float(max(t["best_val_err"] for t in train_outs if t["best_val_err"] is not None))


__all__ = ["SoftMLP", "PathSimplexMLP", "build_model", "train_model",
           "select_matched_checkpoint", "eps_star_from", "LAMBDA_GRID",
           "EVAL_EVERY", "MAX_STEPS"]

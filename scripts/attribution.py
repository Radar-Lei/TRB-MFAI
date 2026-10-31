"""Attribution fields and primary endpoints P1-P4 (frozen definitions).

Objects (contract separation, NEVER cross-checked):
  * Flow-influence field  G[:,j] = IG of f wrt q_j along gamma = q0 -> q1
        G[:,j] = Delta_q_j * int_0^1 J_f(gamma(t)) e_j dt          (flow units)
        Kirchhoff residual R_j = B G[:,j] - M[:,j] Delta_q_j       (P1 target)
  * Outcome apportionment C[:,j] = chain-rule IG of T(f) wrt q_j, decomposed
        C[:,j] = Delta_q_j * int_0^1 diag(grad T(f(gamma(t)))) J_f e_j dt
        completeness: sum_e C[e,j] = IG of T∘f wrt q_j.
    C is never tested against Kirchhoff directly; the responsibility law
    B C[:,j] = M[:,j] beta_j holds UNDER the Level III structural
    conditions (Theorem 5, static form).  The converse is false:
    endpoint balance can also hold accidentally through cancellation
    (main-text accidental-endpoint-conservation example), so an
    endpoint-balanced column is never read as evidence that the
    pointwise structural conditions hold.

Endpoints (relative, dimensionless; unit = (OD j, seed s)):
  P1 = ||R_j|| / (|Delta_q_j| * ||M[:,j]||)
  P2 = sup_t ||B D_t N||_F / ||D_t||_F,  N = orthonormal basis of ker B
       (computed for BOTH outcome classes; zero-threshold <= 1e-6 applies to
        conserving class only)
  P3 = ||B C[:,j] - M[:,j] beta_can_j|| / ||M[:,j] beta_can_j||,
       beta_can_j = Delta_q_j * int alpha_j(t) dt,  alpha_j from L_t = B D_t B^+
       (canonical-witness target residual.  P3 is NOT a pure
        directional residual: with v_j = B C[:,j] and
        beta_end_j = mu_j^T v_j / ||mu_j||^2 the exact split
        ||v_j - mu_j beta_can_j||^2 = ||(I-P_{mu_j}) v_j||^2
                                + ||mu_j||^2 (beta_end_j - beta_can_j)^2
        shows P3 mixes endpoint direction error with reference-scale
        mismatch.  The pure endpoint balance error is reported separately:)
  E_od_end = ||(I - P_{mu_j}) B C[:,j]|| / ||mu_j||
       (relative endpoint same-OD balance error; NaN-safe like P3),
       together with the realized endpoint scale beta_end_j.
  P4 = ||B C[:,j] - int L_t M[:,j] Delta_q_j dt|| / ||int L_t M[:,j] Delta_q_j dt||
       (exact L_t for conserving class; LS-approximate L_t* otherwise, flagged)

All torch computation in float64.  IG quadrature: trapezoid, 200 steps default.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.func import functional_call, jacrev, vmap

TORCH_DTYPE = torch.float64


# ----------------------------------------------------------------------------
# Batched Jacobians along the IG path
# ----------------------------------------------------------------------------
def forward_jacobian(model, X: torch.Tensor) -> torch.Tensor:
    """(B, m, K) Jacobian df/dq via forward-mode layer propagation.
    Supports SoftMLP (plain Sequential), PathSimplexMLP (Sequential + per-OD
    softmax simplex + Delta), their E4 normalized variants (NormSoftMLP /
    NormPathSimplexMLP: same backbones behind standardized q-interfaces; the
    raw-unit Jacobian follows from the chain rule), and CycleAugmentedField.
    Much faster than vmap(jacrev) on CPU for small K."""
    from models import (SoftMLP, PathSimplexMLP, CycleAugmentedField,
                        NormSoftMLP, NormPathSimplexMLP)
    # -- normalized variants first (they subclass the plain models, so the
    #    isinstance checks below would otherwise swallow them) --
    if isinstance(model, NormSoftMLP):
        Xn = (X - model.mu_q) / model.sd_q
        J = _sequential_forward_jac(model.net, Xn)              # df_hat/dq_hat
        return (J * model.sd_f.unsqueeze(-1)) / model.sd_q      # diag(sd_f) J diag(1/sd_q)
    if isinstance(model, NormPathSimplexMLP):
        Xn = (X - model.mu_q) / model.sd_q
        Jz = _sequential_forward_jac(model.net, Xn) / model.sd_q  # dz/dq (B,P,K)
        z = model.net(Xn)                                       # (B, P)
        K = X.shape[1]
        J_h = torch.zeros_like(Jz)
        for j, idx in enumerate(model.od_path_idx):
            zz = z[:, idx]
            zz = zz - zz.max(dim=1, keepdim=True).values
            sig = torch.exp(zz)
            sig = sig / sig.sum(dim=1, keepdim=True)           # (B, n_j)
            Jz_j = Jz[:, idx, :]                               # (B, n_j, K)
            J_sig = Jz_j * sig.unsqueeze(-1) - sig.unsqueeze(-1) * (
                sig.unsqueeze(-1) * Jz_j).sum(dim=1, keepdim=True)
            e_k = torch.zeros(K, dtype=X.dtype)
            e_k[j] = 1.0
            J_h[:, idx, :] = sig.unsqueeze(-1) * e_k + X[:, j:j + 1].unsqueeze(-1) * J_sig
        return torch.einsum("ep,bpk->bek", model.Delta, J_h)
    if isinstance(model, CycleAugmentedField):
        J_inner = forward_jacobian(model.inner, X)              # (B, m, K)
        J_z = _sequential_forward_jac(model.znet, X)            # (B, mu, K)
        return J_inner + torch.einsum("em,bmk->bek", model.N, J_z)
    if isinstance(model, SoftMLP):
        return _sequential_forward_jac(model.net, X)
    if isinstance(model, PathSimplexMLP):
        Jz = _sequential_forward_jac(model.net, X)            # (B, P, K)
        z = model.net(X)                                       # (B, P)
        K = X.shape[1]
        J_h = torch.zeros_like(Jz)
        for j, idx in enumerate(model.od_path_idx):
            zz = z[:, idx]
            zz = zz - zz.max(dim=1, keepdim=True).values
            sig = torch.exp(zz)
            sig = sig / sig.sum(dim=1, keepdim=True)           # (B, n_j)
            Jz_j = Jz[:, idx, :]                               # (B, n_j, K)
            # d sigma / d z = diag(sig) - sig sig^T
            J_sig = Jz_j * sig.unsqueeze(-1) - sig.unsqueeze(-1) * (
                sig.unsqueeze(-1) * Jz_j).sum(dim=1, keepdim=True)
            e_k = torch.zeros(K, dtype=X.dtype)
            e_k[j] = 1.0
            # h_p = q_j * sigma_p  =>  dh_p/dq_k = delta_{jk} sigma_p + q_j J_sig[p,k]
            J_h[:, idx, :] = sig.unsqueeze(-1) * e_k + X[:, j:j + 1].unsqueeze(-1) * J_sig
        return torch.einsum("ep,bpk->bek", model.Delta, J_h)
    # fallback: functorch
    params = {k: v.detach() for k, v in model.named_parameters()}

    def f_single(x):
        return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)

    return vmap(jacrev(f_single))(X)


def _sequential_forward_jac(net, X: torch.Tensor) -> torch.Tensor:
    """Forward-mode Jacobian through a Sequential of Linear/Tanh modules."""
    B, K = X.shape
    J = torch.eye(K, dtype=X.dtype).expand(B, K, K)
    a = X
    for module in net:
        if isinstance(module, torch.nn.Linear):
            J = torch.einsum("oi,bik->bok", module.weight, J)
            a = module(a)
        elif isinstance(module, torch.nn.Tanh):
            t = module(a)                          # tanh(z): z is current pre-activation
            J = (1.0 - t.pow(2)).unsqueeze(-1) * J  # d tanh/dz = 1 - tanh(z)^2
            a = t
        else:
            raise NotImplementedError(f"module {module}")
    return J


def path_points(q0: np.ndarray, q1: np.ndarray, steps: int):
    """Trapezoid nodes t_k = k/steps, k = 0..steps.  Returns X (steps+1, K)."""
    ts = np.linspace(0.0, 1.0, steps + 1)
    dQ = (q1 - q0)[None, :]
    return q0[None, :] + ts[:, None] * dQ, ts, (q1 - q0)


def jacobians_along_path(model, q0, q1, steps=200):
    """J_f(gamma(t_k)) for all k: (steps+1, m, K), plus f at endpoints."""
    X, ts, dQ = path_points(q0, q1, steps)
    Xt = torch.tensor(X, dtype=TORCH_DTYPE)
    J = forward_jacobian(model, Xt)                          # (steps+1, m, K)
    with torch.no_grad():
        f_vals = model(Xt)
    return J.detach().numpy(), f_vals.detach().numpy(), ts, dQ


def grad_T_along_path(T_fn, f_vals: np.ndarray):
    """grad T(f(gamma(t_k))): (steps+1, m).  T_fn: R^m -> R (numpy, differentiable
    analytically -- all our T are polynomials with analytic gradients)."""
    raise NotImplementedError("use specific T classes with analytic grad")


# ----------------------------------------------------------------------------
# Outcome functionals with analytic gradients
# ----------------------------------------------------------------------------
class OutcomeT:
    """Base: T(f) and grad T(f), vectorised over rows; value_torch keeps the
    autograd graph alive for direct-IG reference computations."""

    def value(self, f: np.ndarray) -> np.ndarray: ...
    def grad(self, f: np.ndarray) -> np.ndarray: ...   # same shape as f
    def value_torch(self, f) -> "torch.Tensor": ...

    def __call__(self, f):
        return self.value(f)


class LinearT(OutcomeT):
    """T(f) = w . f  (breaking/conserving built by choosing w)."""

    def __init__(self, w: np.ndarray):
        self.w = np.asarray(w, dtype=float)
        self.wt = torch.tensor(self.w, dtype=TORCH_DTYPE)

    def value(self, f):
        return f @ self.w

    def value_torch(self, f):
        return f @ self.wt

    def grad(self, f):
        return np.broadcast_to(self.w, f.shape)


class ClassAggT(OutcomeT):
    """T(f) = tau(sum_{e in E_k} f_e) over equivalence classes -- the T3c
    'manifold-conserving' family.  tau: scalar -> scalar with analytic derivative.
    Default tau(z) = z^2 (a (sum f)^2-type outcome)."""

    def __init__(self, classes: list[list[int]], tau=None, dtau=None):
        self.classes = [list(c) for c in classes]
        self.tau = tau or (lambda z: z * z)
        self.dtau = dtau or (lambda z: 2.0 * z)

    def value(self, f):
        agg = np.array([f[..., c].sum(axis=-1) for c in self.classes])
        vals = np.array([self.tau(a) for a in agg])
        return vals.sum(axis=0)

    def value_torch(self, f):
        total = None
        for c in self.classes:
            z = f[..., c].sum(dim=-1) ** 2
            total = z if total is None else total + z
        return total

    def grad(self, f):
        g = np.zeros_like(f)
        for c in self.classes:
            agg = f[..., c].sum(axis=-1)
            g[..., c] = np.expand_dims(self.dtau(agg), axis=-1)  # broadcasts over class edges
        return g


# ----------------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------------
def flow_field_G(model, q0, q1, steps=200):
    """G (m, K) and diagnostics: R_j (n, K), sum_j R_j vs r(q1)-r(q0)."""
    J, f_vals, ts, dQ = jacobians_along_path(model, q0, q1, steps)
    # trapezoid integral of J over t: (m, K)
    G = np.zeros((J.shape[1], J.shape[2]))
    for j in range(J.shape[2]):
        G[:, j] = np.trapezoid(J[:, :, j], ts, axis=0) * dQ[j]
    return G, {"f0": f_vals[0], "f1": f_vals[-1], "J_path": J, "ts": ts, "dQ": dQ}


def outcome_field_C(model, T: OutcomeT, q0, q1, steps=200):
    """C (m, K), grad T along path g_t (steps+1, m), plus G internals."""
    J, f_vals, ts, dQ = jacobians_along_path(model, q0, q1, steps)
    gT = T.grad(f_vals)                       # (steps+1, m)
    C = np.zeros((J.shape[1], J.shape[2]))
    for j in range(J.shape[2]):
        C[:, j] = np.trapezoid(gT * J[:, :, j], ts, axis=0) * dQ[j]
    return C, {"gT_path": gT, "f_vals": f_vals, "ts": ts, "dQ": dQ, "J_path": J}


# ----------------------------------------------------------------------------
# Direct autograd IG of T∘f (independent completeness reference)
# ----------------------------------------------------------------------------
def direct_ig_of_T(model, T: OutcomeT, q0, q1, steps=200):
    """IG_j of q -> T(f(q)) computed by full autograd (no chain decomposition).
    Uses midpoint-free trapezoid on the same nodes as C for a tight comparison."""
    X, ts, dQ = path_points(q0, q1, steps)
    Xt = torch.tensor(X, dtype=TORCH_DTYPE, requires_grad=True)
    f = model(Xt)                                   # (steps+1, m)
    Tvals = T.value_torch(f)
    grads = torch.autograd.grad(Tvals.sum(), Xt)[0].detach().numpy()
    return np.array([np.trapezoid(grads[:, j], ts) * dQ[j]
                     for j in range(grads.shape[1])])


# ----------------------------------------------------------------------------
# Shared per-OD endpoint bundle (v18: single source of the scale definitions)
# ----------------------------------------------------------------------------
# v19: relative tolerance declaring the canonical-witness reference target
# mu_j * beta_j^can numerically zero, measured against the data scale of its
# own construction (see endpoint_scales docstring). Round-off accumulation in
# double precision is ~1e-16 per operation; 1e-12 leaves four orders of
# margin while still flagging every analytically-vanishing reference.
REF_ZERO_RTOL = 1e-12


def endpoint_scales(B, M, W_path, J_path, ts, dQ):
    """Canonical-witness and realized endpoint scales for one (q0, q1) path.

    Inputs
      B      : (n, m) incidence matrix
      M      : (n, K) OD indicator columns mu_j
      W_path : (T, m) outcome gradient w(t) = grad T(f(gamma(t)))
      J_path : (T, m, K) flow Jacobian columns dF/dq_j along the path
      ts     : (T,) path parameters in [0, 1]
      dQ     : (K,) demand increments q1 - q0

    Returns dict (all (K,) arrays unless noted)
      beta        canonical-witness scale
                  beta_j = dQ_j * int_0^1 mu_j^T B D_t xhat_j / ||mu_j||^2 dt,
                  xhat_j = B^+ mu_j  (min-norm witness; NEVER the realized
                  Jacobian column -- that path gives beta_end below)
      beta_end    realized endpoint scale  mu_j^T B Ctilde_j / ||mu_j||^2
      E_od_end    pure endpoint same-OD balance error
                  ||(I - P_{mu_j}) B Ctilde_j|| / ||mu_j||
      p3, p3_abs  canonical-witness target residual (relative and absolute)
      reference_norm     ||mu_j * beta_j|| -- the P3 denominator actually used
      reference_tol      data-scale tolerance below which the reference is
                  declared numerically zero (REF_ZERO_RTOL * ||mu_j|| *
                  |dQ_j| * sup_t||w(t)||_inf * ||B||_2 * ||xhat_j||_2)
      reference_near_zero  boolean: the canonical reference target is at
                  round-off scale RELATIVE TO ITS OWN CONSTRUCTION. P3 is
                  then undefined by vanishing reference denominator; the
                  absolute residual p3_abs and the scale reference_norm are
                  stored for reporting. NOTE: this flags the *canonical
                  reference* only -- a near-zero beta_j^can does NOT by
                  itself imply a degenerate realized responsibility field;
                  read beta_end and E_od_end for that (they may be
                  nonzero and same-OD balanced).
      C           (m, K) integrated responsibility columns Ctilde_j =
                  dQ_j * int D_t J_{:,j} dt (trapezoid)
      BC          (n, K) node-space responsibility columns
      alpha_path  (T, K) canonical-witness coefficient along the path
      LtM_int     (n, K) int_0^1 L_t mu_j dt (P4 target integrand)
      r_od_witness_sup  per-OD sup_t ||(I-P_{mu_j}) B D_t xhat_j|| / ||mu_j||
      split_deviation  max_j relative deviation of the orthogonal identity
                  ||v_j - mu_j beta_j||^2 = ||(I-P_{mu_j}) v_j||^2
                                            + ||mu_j||^2 (beta_end_j - beta_j)^2
    """
    W_path = np.asarray(W_path, dtype=float)
    J_path = np.asarray(J_path, dtype=float)
    ts = np.asarray(ts, dtype=float)
    dQ = np.asarray(dQ, dtype=float)
    K = M.shape[1]
    BpM = np.linalg.pinv(B) @ M                    # (m, K)  xhat columns
    Mnorm2 = (M * M).sum(axis=0)                   # (K,)
    Mnorm = np.sqrt(Mnorm2)

    # Data scale of the reference construction mu_j * beta_j^can: the
    # integrand alpha_j(t) is built from w(t), B, and xhat_j, so a target
    # norm below REF_ZERO_RTOL times that scale is round-off of the
    # construction, not a trustworthy denominator (v19). Distinguishes
    # "mathematically zero reference" from "numerically tiny denominator
    # that would inflate P3 to a meaningless ratio".
    Bnorm2 = float(np.linalg.norm(B, 2))
    w_sup = float(np.abs(W_path).max()) if W_path.size else 0.0
    xhat_norm = np.linalg.norm(BpM, axis=0)        # (K,)

    # canonical-witness coefficient along the path: alpha_j(t) uses the
    # min-norm witness xhat_j, not the realized Jacobian column.
    LtMcols = np.stack([B @ (g_t[:, None] * BpM) for g_t in W_path])  # (T, n, K)
    alpha = (M[None] * LtMcols).sum(axis=1) / Mnorm2[None]            # (T, K)
    beta = dQ * np.trapezoid(alpha, ts, axis=0)
    LtM_int = np.trapezoid(LtMcols, ts, axis=0)                      # (n, K)

    # integrated responsibility columns (trapezoid)
    wJ = W_path[:, :, None] * J_path                                 # (T, m, K)
    C = dQ[None, :] * np.trapezoid(wJ, ts, axis=0)                   # (m, K)
    BC = B @ C

    beta_end = (M * BC).sum(axis=0) / Mnorm2
    E_od_end = np.zeros(K)
    p3 = np.full(K, np.nan)
    p3_abs = np.zeros(K)
    reference_norm = np.zeros(K)
    reference_tol = np.zeros(K)
    reference_near_zero = np.zeros(K, dtype=bool)
    r_od_sup = np.zeros(K)
    split_dev = 0.0
    for j in range(K):
        mu = M[:, j]
        v = BC[:, j]
        E_od_end[j] = np.linalg.norm(v - mu * beta_end[j]) / Mnorm[j]
        tgt = mu * beta[j]
        p3_abs[j] = float(np.linalg.norm(v - tgt))
        nrm = np.linalg.norm(tgt)
        reference_norm[j] = nrm
        reference_tol[j] = (REF_ZERO_RTOL * Mnorm[j] * abs(dQ[j])
                            * w_sup * Bnorm2 * xhat_norm[j] + 1e-300)
        reference_near_zero[j] = bool(nrm <= reference_tol[j])
        # P3 is defined only against a reference target distinguishable
        # from round-off: exact zero (nrm == 0) and numerically near-zero
        # (nrm <= reference_tol) both report NaN with the flag stored.
        p3[j] = np.nan if (nrm == 0 or reference_near_zero[j]) \
            else p3_abs[j] / nrm
        # orthogonal decomposition identity (unit-tested in
        # test_endpoint_scales.py): lhs must equal rhs to round-off.
        # Normalisation is by the problem scale ||v||^2 + ||mu beta||^2 --
        # NOT by lhs/rhs themselves, which both collapse to ~1e-32 for a
        # conserving outcome and would turn round-off into an O(1) relative
        # deviation (catastrophic cancellation).
        lhs = p3_abs[j] ** 2
        rhs = (E_od_end[j] * Mnorm[j]) ** 2 \
            + Mnorm2[j] * (beta_end[j] - beta[j]) ** 2
        scale = max(np.linalg.norm(v) ** 2, nrm ** 2, 1e-300)
        split_dev = max(split_dev, abs(lhs - rhs) / scale)
    # pointwise witness residual along the same sweep
    for j in range(K):
        Pj = np.outer(M[:, j], M[:, j]) / Mnorm2[j]
        best = 0.0
        for ti in range(W_path.shape[0]):
            BDx = LtMcols[ti][:, j]
            best = max(best, float(np.linalg.norm(BDx - Pj @ BDx)) / Mnorm[j])
        r_od_sup[j] = best
    # pointwise realized-reference directional residual: the restricted
    # reference-vector condition (I-P_mu) BD_t x_ref = 0 evaluated with the
    # realized Jacobian column as the family representative (Table 5 note:
    # a cycle-invariance pass plus a zero here completes the restricted
    # certification for the realized reference).
    r_od_real = np.zeros(K)
    for j in range(K):
        Pj = np.outer(M[:, j], M[:, j]) / Mnorm2[j]
        best = 0.0
        for ti in range(W_path.shape[0]):
            BDx = B @ (W_path[ti] * J_path[ti][:, j])
            best = max(best, float(np.linalg.norm(BDx - Pj @ BDx)) / Mnorm[j])
        r_od_real[j] = best
    return {"beta": beta, "beta_end": beta_end, "E_od_end": E_od_end,
            "p3": p3, "p3_abs": p3_abs, "C": C, "BC": BC,
            "reference_norm": reference_norm,
            "reference_tol": reference_tol,
            "reference_near_zero": reference_near_zero,
            "alpha_path": alpha, "LtM_int": LtM_int,
            "r_od_witness_sup": r_od_sup,
            "r_od_realized_sup": r_od_real,
            "split_deviation": float(split_dev)}


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
class EndpointCalculator:
    """Pre-factors B, M, N, pinv(B) once per network."""

    def __init__(self, net, ods):
        self.net = net
        self.ods = list(ods)
        self.B = net.B
        self.M = net.od_M(self.ods)
        self.N = net.nullspace_basis()
        self.Bpinv = np.linalg.pinv(self.B)
        self.K = len(self.ods)

    # -- P2 family (path-level) --
    def p2_along_path(self, gT_path: np.ndarray):
        """P2(t) for each trapezoid node; returns (values, sup)."""
        vals = []
        for g_t in gT_path:                    # (m,)
            Mblk = self.B @ (g_t[:, None] * self.N)
            vals.append(np.linalg.norm(Mblk, "fro") / max(np.linalg.norm(g_t), 1e-300))
        vals = np.array(vals)
        return vals, float(vals.max())

    def L_t(self, g_t: np.ndarray):
        """L_t = B D_t B^+  (min-norm LS solution of B D_t = L B).  Only the action
        L_t @ v is needed:  L_t v = B (g_t * (B^+ v))."""
        return lambda v: self.B @ (g_t * (self.Bpinv @ v))

    def alpha_and_LtM(self, gT_path: np.ndarray):
        """Single sweep over path nodes computing both
        (a) alpha_j(t) = <M_j, L_t M_j>/||M_j||^2, then beta inputs,
        (b) L_t M_j columns for the P4 target.
        Uses the pre-factorisation Bpinv @ M (L_t M_j = B (g_t * (B^+ M_j))).
        Returns (alpha_path (T,K), alpha_int (K,), LtM_int (n,K))."""
        T_len = gT_path.shape[0]
        ts = np.linspace(0.0, 1.0, T_len)
        BpM = self.Bpinv @ self.M                        # (n, K) precomputed
        Mnorm = (self.M * self.M).sum(axis=0)            # (K,)
        alphas = np.zeros((T_len, self.K))
        LtMcols = np.zeros((T_len, self.B.shape[0], self.K))
        for ti, g_t in enumerate(gT_path):
            LtMj = self.B @ (g_t[:, None] * BpM)         # (n, K): all j at once
            LtMcols[ti] = LtMj
            alphas[ti] = (self.M * LtMj).sum(axis=0) / Mnorm
        alpha_int = np.trapezoid(alphas, ts, axis=0)
        LtM_int = np.trapezoid(LtMcols, ts, axis=0)
        return alphas, alpha_int, LtM_int

    # -- full endpoint bundle for one (q0, q1) pair --
    def endpoints(self, model, T: OutcomeT, q0, q1, steps=200,
                  with_fields=False):
        dQ = q1 - q0
        G, Ginfo = flow_field_G(model, q0, q1, steps)
        C, Cinfo = outcome_field_C(model, T, q0, q1, steps)
        B, M = self.B, self.M

        r0 = B @ Ginfo["f0"] - M @ q0           # value residuals at endpoints
        r1 = B @ Ginfo["f1"] - M @ q1

        R = B @ G - M * dQ[None, :]              # (n, K) column residuals
        sum_R = R.sum(axis=1)

        # relative endpoints are NaN when the denominator vanishes (e.g. a
        # demand component clipped exactly onto the baseline q0).  NaN values
        # are excluded by nan-aware summaries downstream -- reporting 0 here
        # would fabricate perfect scores.
        p1 = np.full(self.K, np.nan)
        for j in range(self.K):
            denom = abs(dQ[j]) * np.linalg.norm(M[:, j])
            if denom > 0:
                p1[j] = np.linalg.norm(R[:, j]) / denom

        _, p2_sup = self.p2_along_path(Cinfo["gT_path"])
        # v18: single source of the scale family -- the shared bundle
        # computes beta (canonical witness), beta_end, E_od_end, P3 and the
        # orthogonal-split identity in one place for E6/E9/E9b/E12 alike.
        es = endpoint_scales(B, M, Cinfo["gT_path"], Cinfo["J_path"],
                             Cinfo["ts"], dQ)
        alpha, beta = es["alpha_path"], es["beta"]
        p3, beta_end, e_od_end = es["p3"], es["beta_end"], es["E_od_end"]

        p4 = np.full(self.K, np.nan)
        for j in range(self.K):
            tgt = es["LtM_int"][:, j] * dQ[j]
            if np.linalg.norm(tgt) > 0:
                p4[j] = np.linalg.norm(es["BC"][:, j] - tgt) / np.linalg.norm(tgt)

        # IG completeness check: sum_e C[e,j] must equal the *direct* autograd IG
        # of T∘f wrt q_j (independent computation path, no chain decomposition).
        # BOTH sides use the same trapezoid nodes, so this is the CHAIN-RULE
        # IMPLEMENTATION gap only (v19 naming); it shares integration error
        # with the quadrature and does not certify recovery of the true
        # endpoint difference -- that is what endpoint_T_gap measures.
        direct_IG = direct_ig_of_T(model, T, q0, q1, steps)
        ig_T_of_q = C.sum(axis=0)            # sum_e C[e,j] = IG_j of T∘f

        # v19: true-endpoint completeness gaps, independent of shared-grid
        # cancellation:
        #   endpoint_T_gap = | sum_{e,j} C[e,j] - (T(f(q1)) - T(f(q0))) |
        #   endpoint_F_gap = || sum_j G[:,j] - (f(q1) - f(q0)) ||_2
        # (the flow-object vector completeness against the realized field
        # difference; both are the absolute errors, relative forms derived
        # by the callers where a nonzero denominator exists).
        dT = float(T.value(Cinfo["f_vals"][-1]) - T.value(Cinfo["f_vals"][0]))
        endpoint_T_gap = float(abs(ig_T_of_q.sum() - dT))
        endpoint_T_gap_rel = (endpoint_T_gap / abs(dT)) if dT != 0 \
            else float("nan")
        endpoint_F_gap = float(np.linalg.norm(
            G.sum(axis=1) - (Ginfo["f1"] - Ginfo["f0"])))

        out = {
            "P1": p1, "P2": p2_sup, "P3": p3, "P4": p4,
            "E_od_end": e_od_end, "beta_end": beta_end,
            "split_deviation": es["split_deviation"],
            "reference_norm": es["reference_norm"],
            "reference_tol": es["reference_tol"],
            "reference_near_zero": es["reference_near_zero"],
            "r0_norm": float(np.linalg.norm(r0)), "r1_norm": float(np.linalg.norm(r1)),
            "R_columns": R, "sum_R": sum_R,
            "accounting_gap": float(np.linalg.norm(sum_R - (r1 - r0))),
            "completeness_gap": float(np.linalg.norm(ig_T_of_q - direct_IG)),
            "chain_rule_gap": float(np.linalg.norm(ig_T_of_q - direct_IG)),
            "endpoint_T_gap": endpoint_T_gap,
            "endpoint_T_gap_rel": endpoint_T_gap_rel,
            "endpoint_F_gap": endpoint_F_gap,
            "beta": beta, "alpha_path": alpha,
        }
        if with_fields:
            out["G"] = G
            out["C"] = C
        return out

    def p2_for_outcome_class(self, T: OutcomeT, f_samples: np.ndarray):
        """P2 evaluated on arbitrary flow samples (not path-restricted):
        returns sup over rows of ||B D N||_F / ||D||_F."""
        gT = T.grad(f_samples)
        vals = []
        for g_t in gT:
            vals.append(np.linalg.norm(self.B @ (g_t[:, None] * self.N), "fro")
                        / max(np.linalg.norm(g_t), 1e-300))
        return float(np.max(vals))


# ----------------------------------------------------------------------------
# T4 conservation projection (W-weighted, min-norm)
# ----------------------------------------------------------------------------
def project_W(a: np.ndarray, B: np.ndarray, W_diag: np.ndarray, b: np.ndarray):
    """a* = a - W^{-1} B^T (B W^{-1} B^T)^+ (B a - b)  (single column)."""
    Winv = np.diag(1.0 / W_diag)
    K = B @ Winv @ B.T
    corr = Winv @ B.T @ np.linalg.pinv(K) @ (B @ a - b)
    return a - corr, corr


# ----------------------------------------------------------------------------
# Paired statistics (frozen: Cohen's d_z + bootstrap 95% CI, no tests)
# ----------------------------------------------------------------------------
def paired_dz(x: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """x: per-seed paired differences (S,).  Returns dz, ci_low, ci_high."""
    x = np.asarray(x, dtype=float)
    S = len(x)
    dz = x.mean() / (x.std(ddof=1) + 1e-300)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, S, size=S)
        xs = x[idx]
        boots.append(xs.mean() / (xs.std(ddof=1) + 1e-300))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(dz), float(lo), float(hi)


__all__ = ["OutcomeT", "LinearT", "ClassAggT", "flow_field_G", "outcome_field_C",
           "EndpointCalculator", "project_W", "paired_dz", "jacobians_along_path",
           "path_points"]

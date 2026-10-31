"""E9/E10/E11: transportation outcomes, baseline sensitivity, positive-flow
counterexample -- the transportation-facing experiment block.

E9   Common transportation outcome functionals on the Sioux Falls K=8
     snapshot with an exactly conservative (M3-type) field:
       link flow / Beckmann potential / TSTT / emissions / corridor weights /
       uniform aggregate (control).
     For each: pointwise class dispersion, r_cycle, r_OD, endpoint P3, and
     WHICH final class carries the largest violation.

E10  IG baseline sensitivity (zero / mean / low / random-feasible) for a
     passing and a failing outcome: verdict stability, scale drift.

E11  Positive-flow counterexample: f_eps = f_0 + B^+ r_eps with f_0 strictly
     positive and exactly conservative on a compact demand box, r_eps the
     tanh family.  Value residual O(eps) vs flow-attribution residual at
     fixed scale, with min flow > 0 certified on the box.

Outputs: experiments/results/e9_transport.json, e10_baselines.json,
         e11_positive_flow.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nets import (load_sioux_falls, load_sioux_falls_trips,  # noqa: E402
                  sioux_falls_top8_od, shared_cut_loops)
from sue import PathStructure  # noqa: E402
from attribution import (EndpointCalculator, LinearT, OutcomeT,  # noqa: E402
                         flow_field_G, jacobians_along_path)
from run_e1 import analytic_M3_field  # noqa: E402
from run_e6_pointwise import chain_summary  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ZERO_TOL = 1e-9


# ----------------------------------------------------------------------------
# BPR-based transportation outcome functionals (analytic gradients)
# ----------------------------------------------------------------------------
class LinkFlowT(OutcomeT):
    """T(F) = F_{e0}."""

    def __init__(self, e0, m):
        self.e0, self.m = e0, m

    def value(self, f):
        return np.asarray(f)[..., self.e0]

    def value_torch(self, f):
        return f[..., self.e0]

    def grad(self, f):
        g = np.zeros_like(f)
        g[..., self.e0] = 1.0
        return g


class _BPRbase(OutcomeT):
    """Shared BPR machinery: t_e(F) = t0 (1 + 0.15 (F/c)^4)."""

    def __init__(self, t0, cap, m):
        self.t0 = np.asarray(t0, float)
        self.cap = np.asarray(cap, float)
        self.m = m
        self.t0t = torch.tensor(self.t0, dtype=torch.float64)
        self.capt = torch.tensor(self.cap, dtype=torch.float64)

    def _t(self, F):
        x = F / self.cap
        return self.t0 * (1.0 + 0.15 * x ** 4)

    def _t_torch(self, F):
        x = F / self.capt
        return self.t0t * (1.0 + 0.15 * x ** 4)

    def _int_t(self, F):
        # int_0^F t(s) ds = t0 (F + 0.03 F^5 / c^4)
        x = F / self.cap
        return self.t0 * (F + 0.03 * F * x ** 4)


class BeckmannT(_BPRbase):
    """sum_e int_0^{F_e} t_e(s) ds; grad = t_e(F_e)."""

    def value(self, f):
        return self._int_t(np.asarray(f)).sum(-1)

    def value_torch(self, f):
        x = f / self.capt
        return (self.t0t * (f + 0.03 * f * x ** 4)).sum(-1)

    def grad(self, f):
        return self._t(np.asarray(f))


class TSTTT(_BPRbase):
    """sum_e F_e t_e(F_e); grad = t + F t' = t0 (1 + 0.75 x^4), x = F/c
    (t = t0(1+0.15x^4); F t' = t0 * F * 0.6 x^3 / c = 0.6 t0 x^4;
    total 1 + 0.15x^4 + 0.60x^4 = 1 + 0.75 x^4)."""

    def value(self, f):
        F = np.asarray(f)
        x = F / self.cap
        return (self.t0 * F * (1.0 + 0.15 * x ** 4)).sum(-1)

    def value_torch(self, f):
        x = f / self.capt
        return (self.t0t * f * (1.0 + 0.15 * x ** 4)).sum(-1)

    def grad(self, f):
        x = np.asarray(f) / self.cap
        return self.t0 * (1.0 + 0.75 * x ** 4)


class EmissionsT(_BPRbase):
    """E = sum_e kappa_e F_e t_e(F_e) with heterogeneous emission factors
    kappa_e (frozen seed); grad_e = kappa_e t0 (1 + 0.75 x^4)."""

    def __init__(self, t0, cap, m, seed=77):
        super().__init__(t0, cap, m)
        rng = np.random.default_rng(seed)
        self.kappa = rng.uniform(0.5, 2.0, size=m)
        self.kappat = torch.tensor(self.kappa, dtype=torch.float64)

    def value(self, f):
        F = np.asarray(f)
        x = F / self.cap
        return (self.kappa * self.t0 * F * (1.0 + 0.15 * x ** 4)).sum(-1)

    def value_torch(self, f):
        x = f / self.capt
        return (self.kappat * self.t0t * f
                * (1.0 + 0.15 * x ** 4)).sum(-1)

    def grad(self, f):
        x = np.asarray(f) / self.cap
        return self.kappa * self.t0 * (1.0 + 0.75 * x ** 4)


def _sioux_setup():
    net = load_sioux_falls()
    ods = sioux_falls_top8_od()
    demands = load_sioux_falls_trips()
    q_base = np.array([demands[od] for od in ods])
    ps = PathStructure(net, ods)
    eq = net.equivalence_classes(ods)
    classes = [c["edges"] for c in eq["classes"]]
    ec = EndpointCalculator(net, ods)
    return net, ods, q_base, ps, classes, ec


# ----------------------------------------------------------------------------
# E9: transportation outcomes
# ----------------------------------------------------------------------------
def e9():
    net, ods, q_base, ps, classes, ec = _sioux_setup()
    model = analytic_M3_field(ps, 11)
    q0, q1 = 0.2 * q_base, q_base * 1.6
    # v17: config snapshot so the supplementary protocol table (S4) can be
    # checked against the result file verbatim (single seed-11 field on the
    # Sioux Falls frozen path set; the extended network / seeds 21-22 belong
    # to E12 only).
    B, M = net.B, net.od_M(ods)
    ts = np.linspace(0.0, 1.0, 201)
    with torch.no_grad():
        Fpath = model(torch.tensor(np.stack([q0 + t * (q1 - q0) for t in ts]),
                                   dtype=torch.float64)).numpy()
    cons_max = float(np.linalg.norm(Fpath @ B.T
                                    - np.stack([q0 + t * (q1 - q0) for t in ts]) @ M.T,
                                    axis=1).max())
    # corridor weights: the LONGEST frozen path of OD 0 (paper OD (10,16)),
    # 6 arcs -- a genuine multi-arc corridor (v7: the first frozen path is
    # the direct single arc and would duplicate the single-link control)
    i_corr = max(ps.od_path_idx[0],
                 key=lambda i: (len(ps.paths[i]), -ps.free_flow_cost[i]))
    corridor = np.zeros(net.m)
    corridor[ps.paths[i_corr]] = 1.0
    # v17 active-link control: the single-link outcome above sits on arc
    # 1->2, which lies on NONE of the 80 frozen paths (vacuous for the
    # frozen-path fields).  Select a link that IS frozen-path-active and
    # has nonzero response to a query OD: the middle arc of the first
    # (free-flow-shortest) frozen path of OD 0.  Non-hollow counterpart of
    # the inactive-link outcome; no new network, training, or grids.
    p0 = ps.paths[ps.od_path_idx[0][0]]
    e_act = p0[len(p0) // 2]
    assert any(e_act in ps.paths[i] for i in range(len(ps.paths))), \
        "active-link arc must lie on the frozen path set"
    Ts = {
        "link_flow": LinkFlowT(0, net.m),
        "link_flow_active": LinkFlowT(e_act, net.m),
        "beckmann": BeckmannT(net.free_flow_time, net.capacity, net.m),
        "tstt": TSTTT(net.free_flow_time, net.capacity, net.m),
        "emissions": EmissionsT(net.free_flow_time, net.capacity, net.m),
        "corridor_fixed_weights": LinearT(corridor),
        "uniform_aggregate_CONTROL": LinearT(np.ones(net.m)),
    }
    _ACTIVE_LINK = {"edge_index": int(e_act),
                    "edge": tuple(int(x) for x in net.edges[e_act]),
                    "on_frozen_paths": int(sum(e_act in ps.paths[i]
                                               for i in range(len(ps.paths))))}
    report = {"config": {"analytic_field": "analytic_M3_field(ps, 11)",
                         "construction_seed": 11,
                         "network": "sioux_falls_24n_76m (not the E12 extended net)",
                         "frozen_paths": len(ps.paths),
                         "active_link_outcome": _ACTIVE_LINK,
                         "eval_grid_nodes": 201,
                         "field_conservation_max": cons_max},
              "n_classes": len(classes),
              "class_sizes": [len(c) for c in classes],
              "m": net.m}
    # analytic-gradient spot check vs torch autograd (round-1 review item):
    # max |analytic grad - autograd grad| at the realized base flow
    with torch.no_grad():
        qm = torch.tensor(0.5 * (q0 + q1), dtype=torch.float64)[None, :]
        f_ref = model(qm).numpy()[0]
    grad_checks = {}
    for name in ("beckmann", "tstt", "emissions"):
        T = Ts[name]
        f_t = torch.tensor(f_ref, dtype=torch.float64, requires_grad=True)
        T.value_torch(f_t).sum().backward()
        ga = T.grad(f_ref)
        gt = f_t.grad.numpy()
        grad_checks[name] = {
            "max_abs_diff": float(np.abs(ga - gt).max()),
            "analytic_range": [float(ga.min()), float(ga.max())],
            "autograd_range": [float(gt.min()), float(gt.max())],
        }
        print(f"  E9 grad check {name}: max_abs_diff="
              f"{grad_checks[name]['max_abs_diff']:.3e}", flush=True)
    report["grad_autograd_check"] = grad_checks
    for name, T in Ts.items():
        s = chain_summary(ec, model, T, q0, q1, classes)
        # failing-class localization: argmax over per-class dispersion
        per_class = s["sup_r_class_per_class"]
        worst = int(np.argmax(per_class)) if per_class else 0
        s["worst_class"] = worst
        s["worst_class_size"] = len(classes[worst]) if classes else 0
        s["law_verdict"] = bool(s["sup_r_class"] < ZERO_TOL)
        report[name] = s
        print(f"  E9 {name:28s} verdict={s['law_verdict']} "
              f"r_class={s['sup_r_class']:.3e} r_od={s['sup_r_od']:.3e} "
              f"P3={s['endpoint_p3_median']:.3e} worst_class={worst}",
              flush=True)
    return report


# ----------------------------------------------------------------------------
# E10: IG baseline sensitivity
# ----------------------------------------------------------------------------
def e10():
    net, ods, q_base, ps, classes, ec = _sioux_setup()
    model = analytic_M3_field(ps, 11)
    rng = np.random.default_rng(123)
    q1 = q_base * 1.6
    baselines = {
        "zero": np.zeros_like(q_base),
        "mean_demand": 0.9 * q_base,
        "low_demand": 0.2 * q_base,
        "random_feasible": np.clip(rng.uniform(0.1, 1.5, size=len(ods)), 0.1, 1.5) * q_base,
    }
    Ts = {
        "uniform_aggregate_CONTROL": LinearT(np.ones(net.m)),
        "tstt": TSTTT(net.free_flow_time, net.capacity, net.m),
    }
    report = {}
    for tname, T in Ts.items():
        rows = {}
        for bname, q0 in baselines.items():
            s = chain_summary(ec, model, T, q0, q1, classes, steps=200)
            # flow-attribution endpoint P1 for scale context
            G, _ = flow_field_G(model, q0, q1, 200)
            dQ = q1 - q0
            p1s = []
            for j in range(ec.K):
                denom = abs(dQ[j]) * np.linalg.norm(ec.M[:, j])
                if denom > 0:
                    p1s.append(np.linalg.norm(
                        ec.B @ G[:, j] - ec.M[:, j] * dQ[j]) / denom)
            rows[bname] = {"sup_r_cycle": s["sup_r_cycle"],
                           "sup_r_od": s["sup_r_od"],
                           "sup_r_class": s["sup_r_class"],
                           "endpoint_p3_median": s["endpoint_p3_median"],
                           "flow_p1_median": float(np.median(p1s)),
                           "verdict_class": bool(s["sup_r_class"] < ZERO_TOL)}
        verdicts = {b: r["verdict_class"] for b, r in rows.items()}
        report[tname] = {"rows": rows,
                         "verdict_stable_across_baselines":
                             bool(len(set(verdicts.values())) == 1)}
        print(f"  E10 {tname}: verdict stable = "
              f"{report[tname]['verdict_stable_across_baselines']}", flush=True)
    return report


# ----------------------------------------------------------------------------
# E11: positive-flow counterexample
# ----------------------------------------------------------------------------
class E11Base(torch.nn.Module):
    """Analytic strictly positive exactly conservative field on the
    two-path-plus-bridge network (OD1 = 0->3 two paths, OD2 = 4->5 bridge):
        f0(q) = (sigma q1, sigma q1, (1-sigma) q1, (1-sigma) q1, q2),
        sigma(q) = 0.4 + 0.2 tanh(q1)  in (0.2, 0.6) strictly.
    Positivity on the box [0.2,1.6]^2 has the closed-form margin
    min(0.2*0.2, 0.2) = 0.04; conservation holds by path decomposition."""

    def forward(self, q):
        s = 0.4 + 0.2 * torch.tanh(q[:, 0])
        return torch.stack([s * q[:, 0], s * q[:, 0],
                            (1 - s) * q[:, 0], (1 - s) * q[:, 0],
                            q[:, 1]], dim=1)


class PositivePerturbedField(torch.nn.Module):
    """f_eps(q) = f_0(q) + B^+ r_eps(q), r_eps = eps v tanh((q1-q2)/eps)."""

    def __init__(self, f0, Bpinv, v, eps):
        super().__init__()
        self.f0 = f0
        self.Bp = torch.tensor(Bpinv, dtype=torch.float64)
        self.v = torch.tensor(v, dtype=torch.float64)
        self.eps = eps

    def forward(self, q):
        s = self.eps * torch.tanh((q[:, 0] - q[:, 1]) / self.eps)
        r = s[:, None] * self.v[None, :]
        return self.f0(q) + r @ self.Bp.T


def e11():
    from nets import Network
    net = Network("e11_two_path", [(0, 1), (1, 3), (0, 2), (2, 3), (4, 5)])
    ods = [(0, 3), (4, 5)]
    ec = EndpointCalculator(net, ods)
    B = net.B
    rng = np.random.default_rng(5)
    v = rng.normal(size=B.shape[0])
    # v in im(B) = per-connected-component zero-sum subspace (the network has
    # two components: {0,1,2,3} and {4,5})
    for comp in ([0, 1, 2, 3], [4, 5]):
        v[comp] -= v[comp].mean()
    v /= np.linalg.norm(v)
    Bpinv = np.linalg.pinv(B)
    f0 = E11Base()

    q_lo, q_hi = 0.2, 1.6
    grid = np.stack(np.meshgrid(np.linspace(q_lo, q_hi, 21),
                                np.linspace(q_lo, q_hi, 21)), -1).reshape(-1, 2)
    with torch.no_grad():
        f0_grid = f0(torch.tensor(grid, dtype=torch.float64)).numpy()
    margin0 = float(f0_grid.min(axis=None))
    assert margin0 > 0

    q0 = np.array([q_lo, q_lo])
    q1 = np.array([q_hi, q_hi])          # diagonal parameter path
    rows = []
    for eps in (1e-2, 1e-3, 1e-4, 1e-5):
        model = PositivePerturbedField(f0, Bpinv, v, eps)
        with torch.no_grad():
            f_grid = model(torch.tensor(grid, dtype=torch.float64)).numpy()
        val_res = np.linalg.norm(B @ f_grid.T - ec.M @ grid.T, axis=0)
        # flow-attribution endpoint residual (P1-style) along the diagonal
        G, info = flow_field_G(model, q0, q1, steps=200)
        dQ = q1 - q0
        p1s = []
        for j in range(ec.K):
            denom = abs(dQ[j]) * np.linalg.norm(ec.M[:, j])
            p1s.append(np.linalg.norm(ec.B @ G[:, j] - ec.M[:, j] * dQ[j]) / denom)
        row = {"eps": eps,
               "sup_value_residual_grid": float(val_res.max()),
               "min_flow_grid": float(f_grid.min()),
               "strictly_positive": bool(f_grid.min() > 0),
               "flow_attribution_p1": [float(x) for x in p1s],
               "f0_margin": margin0}
        rows.append(row)
        print(f"  E11 eps={eps:g}: val_res={row['sup_value_residual_grid']:.3e} "
              f"min_flow={row['min_flow_grid']:.4f} P1={p1s}", flush=True)
    out = {"f0_margin_on_box": margin0, "rows": rows,
           "attribution_residual_flat": bool(
               len({round(r["flow_attribution_p1"][0], 2)
                    for r in rows}) == 1)}
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("E9: transportation outcomes on Sioux Falls ...", flush=True)
    e9r = e9()
    with open(os.path.join(RESULTS, "e9_transport.json"), "w") as fh:
        json.dump(e9r, fh, indent=1, default=float)
    print("E10: baseline sensitivity ...", flush=True)
    e10r = e10()
    with open(os.path.join(RESULTS, "e10_baselines.json"), "w") as fh:
        json.dump(e10r, fh, indent=1, default=float)
    print("E11: positive-flow counterexample ...", flush=True)
    e11r = e11()
    with open(os.path.join(RESULTS, "e11_positive_flow.json"), "w") as fh:
        json.dump(e11r, fh, indent=1, default=float)
    print("written e9/e10/e11 result files", flush=True)


if __name__ == "__main__":
    main()

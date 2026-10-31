"""Shared training/evaluation pipeline for E2 (pilot v2) and E3 (confirmatory).

run_suite(net, ods, theta, ...):
  1. path structure, dataset generation (disk-cached npz), frozen 70/15/15 split
     with DYNAMIC boundaries (n_train may be budget-downgraded to 2500)
  2. train M1 (lambda grid) / M2 / M3 x seeds -- optionally in a process pool
  3. error matching: eps* aggregated BY METHOD (M1 = min over its lambda grid,
     M4 inherits M1@lam=100), runs entering [0.95 eps*, 1.05 eps*] use the first
     checkpoint in band; others exit the head-to-head comparison and enter the
     (val error x P1) Pareto secondary evidence
  4. endpoint evaluation P1-P4 on frozen test points for each outcome class;
     NaN-safe (zero denominators from clipped demands are excluded, not zeroed)
  5. M4 post-hoc projection of M1@lam=100 flow fields per seed, on the SAME
     evaluation points, reusing the G fields (projection is closed-form);
     records the T4 completeness shift and the endpoint-conservation premise
  6. IG step-count convergence appendix (60/100/200/400)

Outputs: experiments/results/{tag}_theta{theta}.json + {tag}_theta{theta}_raw.npz
(raw arrays keyed '{method}|{kw}|{seed}|{outcome}|{field}' -- flat, no pickles)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nets import Network  # noqa: E402
from sue import (PathStructure, generate_dataset, split_indices,  # noqa: E402
                 in_domain_baseline, N_TRAIN, N_VAL, N_TEST)
from models import (train_model, select_matched_checkpoint, build_model,  # noqa: E402
                    LAMBDA_GRID, M2_MU)
from attribution import (EndpointCalculator, LinearT, ClassAggT,  # noqa: E402
                         project_W)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ----------------------------------------------------------------------------
# dataset with disk cache
# ----------------------------------------------------------------------------
def get_dataset(ps: PathStructure, q_base, theta, n_total, seed, tol, tag):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"cache_{tag}_theta{theta}_n{n_total}_tol{tol:g}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {"q": z["q"], "F": z["F"], "residuals": z["residuals"],
                "cons_err": float(z["cons_err"]), "theta": theta, "tol": tol}
    print(f"  generating dataset {tag} theta={theta} n={n_total} tol={tol:g}", flush=True)
    t0 = time.time()
    d = generate_dataset(ps, q_base, theta, n_total, seed, tol=tol)
    print(f"  dataset done in {time.time()-t0:.0f}s (mean iters "
          f"{d['mean_iters']:.0f}, max SUE res {d['residuals'].max():.1e}, "
          f"fails>tol: {d['n_fail']})", flush=True)
    np.savez_compressed(path, q=d["q"], F=d["F"], residuals=d["residuals"],
                        cons_err=d["cons_err"])
    return d


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def _serial(ps, data, split, jobs, threads_per=4):
    torch.set_num_threads(threads_per)
    res = {}
    for method, kw, seed in jobs:
        t0 = time.time()
        out = train_model(method, ps, data, split, seed, **dict(kw))
        out["train_seconds"] = time.time() - t0
        res[f"{method}|{kw}|{seed}"] = out
        print(f"    {method}{kw} seed={seed}: {out['train_seconds']:.0f}s "
              f"best_val={out['best_val_err']:.3e} ({out['stopped_by']})", flush=True)
    return res


def _pool_train(payload):
    job, ps, data, split, threads_per = payload
    method, kw, seed = job
    torch.set_num_threads(threads_per)
    t0 = time.time()
    out = train_model(method, ps, data, split, seed, **dict(kw))
    out["train_seconds"] = time.time() - t0
    return (f"{method}|{kw}|{seed}", out)


# ----------------------------------------------------------------------------
# eps* aggregation BY METHOD (frozen: four methods; M4 inherits M1@lam=100)
# ----------------------------------------------------------------------------
def eps_star_by_method(train_outs: dict):
    groups = {}
    for key, out in train_outs.items():
        method, kw = key.split("|")[0], key.split("|")[1]
        if method == "M1":
            groups.setdefault("M1", []).append(out["best_val_err"])
            if "'lam', 100.0" in kw.replace(",)", ")") or "'lam', 100.0)" in kw:
                groups.setdefault("M4", []).append(out["best_val_err"])
        elif method in ("M2", "M3"):
            groups.setdefault(method, []).append(out["best_val_err"])
    mins = {g: float(min(v)) for g, v in groups.items()}
    return float(max(mins.values())), mins


# ----------------------------------------------------------------------------
# evaluation (single G integration reused for M4 projections)
# ----------------------------------------------------------------------------
def evaluate_model(ec: EndpointCalculator, model, T, q0, q1_points, steps=200,
                   project_specs=None):
    """project_specs: list of (name, W_diag).  Projected P1 (M4) reuses the G
    field from the same integration -- no second IG pass.  NaN-safe."""
    P1s, P3s, P4s, P2s, EodEnds, BetaEnds, comp_gaps, acc_gaps, r0s, r1s = [], [], [], [], [], [], [], [], [], []
    proj_p1 = {name: [] for name, _ in (project_specs or [])}
    proj_shift = {name: [] for name, _ in (project_specs or [])}
    proj_premise = {name: [] for name, _ in (project_specs or [])}
    B, M = ec.B, ec.M
    for q1 in q1_points:
        o = ec.endpoints(model, T, q0, q1, steps=steps, with_fields=True)
        P1s.append(o["P1"]); P3s.append(o["P3"]); P4s.append(o["P4"]); P2s.append(o["P2"])
        EodEnds.append(o.get("E_od_end", np.full(ec.K, np.nan)))
        BetaEnds.append(o.get("beta_end", np.full(ec.K, np.nan)))
        comp_gaps.append(o["completeness_gap"]); acc_gaps.append(o["accounting_gap"])
        r0s.append(o["r0_norm"]); r1s.append(o["r1_norm"])
        if project_specs:
            G, dQ = o["G"], q1 - q0
            G_total = G.sum(axis=1)                     # (m,) total attribution
            b_total = M @ dQ                            # (n,) conserved endpoint total
            premise_gap = float(np.linalg.norm(B @ G_total - b_total))
            for name, Wd in project_specs:
                p1p = np.full(ec.K, np.nan)
                Gp_total = np.zeros_like(G_total)
                for j in range(ec.K):
                    Gj, _ = project_W(G[:, j], B, Wd, M[:, j] * dQ[j])
                    Gp_total += Gj
                    denom = abs(dQ[j]) * np.linalg.norm(M[:, j])
                    if denom > 0:
                        p1p[j] = np.linalg.norm(B @ Gj - M[:, j] * dQ[j]) / denom
                proj_p1[name].append(p1p)
                # T4 completeness bookkeeping: premise violation BEFORE projection
                # and the shift the projection induces in the TOTAL attribution.
                proj_premise[name].append(premise_gap)
                proj_shift[name].append(float(np.linalg.norm(Gp_total - G_total)))
    out = {"P1": np.array(P1s), "P3": np.array(P3s), "P4": np.array(P4s),
           "P2": np.array(P2s), "E_od_end": np.array(EodEnds),
           "beta_end": np.array(BetaEnds),
           "completeness_gap": np.array(comp_gaps),
           "accounting_gap": np.array(acc_gaps),
           "r0_norm": np.array(r0s), "r1_norm": np.array(r1s)}
    for name in proj_p1:
        out[f"{name}__P1_projected"] = np.array(proj_p1[name])
        out[f"{name}__completeness_shift"] = np.array(proj_shift[name])
        out[f"{name}__premise_violation"] = np.array(proj_premise[name])
    return out


def outcome_classes(net, ods):
    """Frozen outcome constructions for H3: conserving = class-constant w;
    breaking = heterogeneous w inside the largest class CROSSED BY an OD path."""
    eq = net.equivalence_classes(ods)
    classes = [c["edges"] for c in eq["classes"]]
    crossed = {a for ks in eq["od_atoms"].values() for a in ks}
    w_cons = np.zeros(net.m)
    for k, c in enumerate(classes):
        w_cons[list(c)] = 1.0 + 0.7 * k
    w_break = w_cons.copy()
    multi = [c for i, c in enumerate(classes) if len(c) > 1 and i in crossed]
    tgt = max(multi, key=len) if multi else max(classes, key=len)
    rng = np.random.default_rng(20260820)
    w_break[list(tgt)] += rng.uniform(0.3, 0.9, size=len(tgt))
    return {"conserving": LinearT(w_cons), "breaking": LinearT(w_break),
            # v17 name: nonlinear class-aggregate outcome (no per-update re-projection)
            "manifold": ClassAggT(classes), "classes": classes, "target": tgt}


def _nanmed(a):
    return float(np.nanmedian(a)) if np.size(a) else float("nan")


def run_suite(tag, net: Network, ods, theta, tol, n_train, seeds=(0, 1, 2, 3, 4),
              n_eval=100, ig_steps=200, use_pool=False, n_workers=8,
              threads_per=4, q_base=None):
    t_start = time.time()
    ps = PathStructure(net, ods)
    n_total = n_train + N_VAL + N_TEST
    if q_base is None:
        q_base = np.ones(len(ods))
    data = get_dataset(ps, q_base, theta, n_total, seed=20260820, tol=tol, tag=tag)
    tr, va, te = split_indices(n_total, n_train=n_train)
    split = {"train": tr, "val": va, "test": te}
    print(f"[{tag} theta={theta}] P={ps.P}, dataset cons_err={data['cons_err']:.1e}, "
          f"split {len(tr)}/{len(va)}/{len(te)}", flush=True)

    # ---- training ----
    t0 = time.time()
    jobs = []
    for seed in seeds:
        for lam in LAMBDA_GRID:
            jobs.append(("M1", (("lam", lam),), seed))
        for mu in (M2_MU, 1.0):          # mu=100 main + mu=1 sensitivity (appendix)
            jobs.append(("M2", (("mu", mu),), seed))
        jobs.append(("M3", (), seed))
    if use_pool:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            outs = pool.map(_pool_train,
                            [(j, ps, data, split, threads_per) for j in jobs])
        train_outs = dict(outs)
    else:
        train_outs = _serial(ps, data, split, jobs, threads_per)
    print(f"  training done in {time.time()-t0:.0f}s", flush=True)

    # ---- error matching (eps* by method) ----
    eps_star, mins = eps_star_by_method(train_outs)
    matched = {}
    for key, out in train_outs.items():
        ck, ok, step, val_sel = select_matched_checkpoint(out, eps_star)
        matched[key] = {"state": ck, "in_band": ok, "step": step,
                        "selected_val_err": val_sel,
                        "best_val_err": out["best_val_err"]}
    print(f"  eps*={eps_star:.3e} (per-method minima {mins})", flush=True)

    # ---- evaluation ----
    torch.set_num_threads(threads_per)
    ec = EndpointCalculator(net, ods)
    q0 = in_domain_baseline(q_base)
    eval_idx = te[:n_eval]
    q1_points = data["q"][eval_idx]
    Ts = outcome_classes(net, ods)

    summary = {"tag": tag, "theta": theta, "tol": tol, "n_train": int(len(tr)),
               "P": ps.P, "eps_star": eps_star, "eps_star_by_method": mins,
               "seeds": list(seeds), "ig_steps": ig_steps, "n_eval": int(n_eval),
               "outcome_classes": [list(map(int, c)) for c in Ts["classes"]],
               "breaking_target": list(map(int, Ts["target"])),
               "methods": {}, "pareto": []}
    raw = {}
    for key in sorted(matched):
        method, kw_str, seed = key.split("|")
        kw = dict(eval(f"dict({kw_str})")) if kw_str else {}
        model = build_model(method, ps, int(seed))
        model.load_state_dict(matched[key]["state"])
        specs = None
        if method == "M1" and kw.get("lam") == 100.0:
            specs = [("M4_W_I", np.ones(net.m)), ("M4_W_cap", net.capacity)]
        entry = {"in_band": matched[key]["in_band"],
                 "match_step": matched[key]["step"],
                 "selected_val_err": matched[key]["selected_val_err"],
                 "best_val_err": matched[key]["best_val_err"]}
        for tname, T in (("conserving", Ts["conserving"]), ("breaking", Ts["breaking"])):
            ev = evaluate_model(ec, model, T, q0, q1_points, steps=ig_steps,
                                project_specs=specs)
            for field in ("P1", "P3", "P4", "P2", "completeness_gap",
                          "accounting_gap", "r0_norm", "r1_norm",
                          "E_od_end", "beta_end"):
                raw[f"{key}|{tname}|{field}"] = ev[field]
            for extra in (specs or []):
                for field in ("P1_projected", "completeness_shift",
                              "premise_violation"):
                    raw[f"{key}|{tname}|{extra[0]}__{field}"] = ev[f"{extra[0]}__{field}"]
            entry[tname] = {
                "P1_median": _nanmed(ev["P1"]),
                "P1_iqr": float(np.nanpercentile(ev["P1"], 75) - np.nanpercentile(ev["P1"], 25)),
                "P2_median": _nanmed(ev["P2"]),
                "P3_median": _nanmed(ev["P3"]),
                "P3_iqr": float(np.nanpercentile(ev["P3"], 75) - np.nanpercentile(ev["P3"], 25)),
                "P4_median": _nanmed(ev["P4"]),
                "P4_iqr": float(np.nanpercentile(ev["P4"], 75) - np.nanpercentile(ev["P4"], 25)),
                "accounting_gap_max": float(np.max(ev["accounting_gap"])),
                "comp_gap_max": float(np.max(ev["completeness_gap"])),
                "r0_norm_median": _nanmed(ev["r0_norm"]),
                "r1_norm_median": _nanmed(ev["r1_norm"]),
                "E_od_end_median": _nanmed(ev["E_od_end"]),
                "beta_end_median": _nanmed(ev["beta_end"]),
            }
            if specs:
                for wname, _ in specs:
                    entry[tname][f"{wname}_P1_projected_median"] = _nanmed(ev[f"{wname}__P1_projected"])
                    entry[tname][f"{wname}_P1_projected_max"] = float(np.nanmax(ev[f"{wname}__P1_projected"]))
                    entry[tname][f"{wname}_completeness_shift_median"] = _nanmed(ev[f"{wname}__completeness_shift"])
                    entry[tname][f"{wname}_premise_violation_median"] = _nanmed(ev[f"{wname}__premise_violation"])
        summary["methods"][key] = entry
        summary["pareto"].append({"run": key,
                                  "val_err": matched[key]["selected_val_err"],
                                  "in_band": matched[key]["in_band"],
                                  "P1_conserv": entry["conserving"]["P1_median"]})
        print(f"  eval {key}: in_band={entry['in_band']} "
              f"P1_cons={entry['conserving']['P1_median']:.2e} "
              f"P3_cons={entry['conserving']['P3_median']:.2e} "
              f"P3_break={entry['breaking']['P3_median']:.2e}", flush=True)

    # ---- IG step convergence appendix (M3 seed0, conserving, 10 points) ----
    key0 = f"M3|()|{seeds[0]}"
    model0 = build_model("M3", ps, seeds[0])
    model0.load_state_dict(matched[key0]["state"])
    ig_conv = {}
    for steps in (60, 100, 200, 400):
        ev = evaluate_model(ec, model0, Ts["conserving"], q0, q1_points[:10],
                            steps=steps)
        ig_conv[steps] = {"P1_median": _nanmed(ev["P1"]),
                          "P3_median": _nanmed(ev["P3"])}
    summary["ig_step_convergence"] = ig_conv
    summary["wall_seconds"] = time.time() - t_start

    with open(os.path.join(RESULTS, f"{tag}_theta{theta}.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(RESULTS, f"{tag}_theta{theta}_raw.npz"),
                        **{k: np.asarray(v, dtype=float) for k, v in raw.items()})
    return summary

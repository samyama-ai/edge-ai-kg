"""Deterministic synthetic generator for the Edge AI deployment fleet.

Everything this module emits is SYNTHETIC. Vendor, board and SoC names are
deliberately fictional so no number here can be mistaken for a claim about a
real product. The only real data in this KG is the ONNX operator catalog
(see `etl/onnx_catalog.py`).

Why synthetic: the interesting questions in edge-AI deployment are structural
("which operators have no kernel on this accelerator, and what does the CPU
fallback cost me?"). Those need dense, complete coverage across a hardware
fleet -- which no public dataset provides, and which cannot be honestly faked
using real product names.

Determinism: everything derives from `--seed` (default 20260814), so the same
seed reproduces the same graph byte-for-byte.

Metrics are *derived*, not invented -- see `docs/data-provenance.md` for the
cost model and its stated limits.
"""
from __future__ import annotations

import random
from itertools import pairwise

from etl import sites as sites_mod
from etl.archetypes import (
    ACCEL_ARCHETYPES,
    CERTIFICATIONS,
    CLINICAL_TASKS,
    DATASETS,
    FORM_FACTORS,
    MODEL_FAMILIES,
    PRECISIONS,
    RUNTIMES,
    SENSORS,
    SIGNAL_STAGES,
    VENDORS,
)

# Re-exported, not moved away: every caller in the repo imports `Fleet`,
# `write`, `load` and the cache types from here, so the split cost no call
# sites. `etl/fleet.py` holds them because nothing in it draws a random
# number, which is the seam this file was split on.
from etl.fleet import (
    DATA_DIR,
    FLEET_PATH,
    GENERATED_LABELS,
    Fleet,
    StaleFleetCache,
    _rid,
    load,
    write,
)
from etl.onnx_catalog import Operator, load_cached

__all__ = ["DATA_DIR", "DEFAULT_SEED", "FLEET_PATH", "GENERATED_LABELS",
           "Fleet", "StaleFleetCache", "generate", "load", "write"]

DEFAULT_SEED = 20260814

# Deployment latency is the cost model's output times a spread, rounded.
# Named so tests can assert against the generator rather than restate it.
LATENCY_JITTER = (1.05, 1.45)
LATENCY_DECIMALS = 3
# `fallback_fraction` is stored rounded while `latency_ms` is computed from the
# unrounded value, so a test recomputing the cost model from the graph carries
# this error too. Named for the same reason as the two above.
FALLBACK_FRACTION_DECIMALS = 4


def generate(seed: int = DEFAULT_SEED, scale: float = 1.0,
             operators: list[Operator] | None = None) -> Fleet:
    rng = random.Random(seed)
    ops = operators if operators is not None else load_cached()
    fleet = Fleet(seed=seed, scale=scale)

    def n(base: int) -> int:
        return max(1, round(base * scale))

    # ---------------- Vendors ----------------
    vendors = []
    for i, (name, country) in enumerate(VENDORS):
        vid = _rid("vendor", i)
        vendors.append({"id": vid, "name": name, "country": country})
    fleet.add_nodes("Vendor", vendors)

    # ---------------- Operators (REAL) ----------------
    op_rows = [{
        "id": o.id, "name": o.name, "domain": o.domain,
        "since_version": o.since_version, "version_count": o.version_count,
        "category": o.category, "is_control_flow": 1 if o.is_control_flow else 0,
    } for o in ops]
    # The ONNX operator catalog is REAL public data (Apache-2.0, onnx/onnx) --
    # it is the one part of this generator that is not invented, so it must not
    # be stamped synthetic.
    fleet.add_nodes("Operator", op_rows, provenance="real", source="onnx")
    ops_by_cat: dict[str, list[Operator]] = {}
    for o in ops:
        ops_by_cat.setdefault(o.category, []).append(o)

    # ---------------- Certifications ----------------
    certs = [{"id": _rid("cert", i), "name": nm, "body": body, "class": cls}
             for i, (nm, body, cls) in enumerate(CERTIFICATIONS)]
    fleet.add_nodes("Certification", certs)

    # ---------------- Sites (#34) ----------------
    # The shape, and the decision behind it, are in `etl/sites.py` and
    # `docs/location-scope.md`. Generated-layer only: the real layer is never
    # placed.
    # Its own stream, derived from the run's seed so placement is still
    # deterministic, and separate so it cannot perturb the draws above.
    site_rng = random.Random(seed + 34)
    sites = sites_mod.build_sites(n(len(sites_mod.SITE_SUFFIXES)), _rid)
    fleet.add_nodes("Site", sites)

    # ---------------- Datasets ----------------
    datasets = [{"id": _rid("dataset", i), "name": nm, "source": src,
                 "subjects": subj, "hours": hrs, "license": lic}
                for i, (nm, src, subj, hrs, lic) in enumerate(DATASETS)]
    fleet.add_nodes("Dataset", datasets)

    # ---------------- Sensors ----------------
    sensors = [{"id": _rid("sensor", i), "name": nm, "modality": mod,
                "sample_rate_hz": sr, "channels": ch, "adc_bits": bits}
               for i, (nm, mod, sr, ch, bits) in enumerate(SENSORS)]
    fleet.add_nodes("Sensor", sensors)

    # ---------------- Signal stages ----------------
    stages = []
    for i, (nm, kind, win, kmacs, cats) in enumerate(SIGNAL_STAGES):
        stages.append({"id": _rid("stage", i), "name": nm, "kind": kind,
                       "window_ms": win, "cost_kmacs": kmacs, "_cats": cats})
    fleet.add_nodes("SignalStage", [{k: v for k, v in s.items() if not k.startswith("_")}
                                    for s in stages])
    for s in stages:
        for cat in s["_cats"]:
            for o in rng.sample(ops_by_cat.get(cat, []),
                                min(2, len(ops_by_cat.get(cat, [])))):
                fleet.add_edge("SignalStage", s["id"], "USES_OPERATOR", "Operator", o.id, None)

    # ---------------- Runtimes ----------------
    runtimes = []
    for i, (nm, ver, fmt, kinds) in enumerate(RUNTIMES):
        runtimes.append({"id": _rid("runtime", i), "name": nm, "version": ver,
                         "format": fmt, "_kinds": kinds})
    fleet.add_nodes("Runtime", [{k: v for k, v in r.items() if not k.startswith("_")}
                                for r in runtimes])

    # ---------------- SoCs + Accelerators ----------------
    socs, accelerators = [], []
    for i in range(n(40)):
        vendor = rng.choice(vendors)
        soc = {
            "id": _rid("soc", i),
            "name": f"{vendor['name'].split()[0][:3].upper()}-S{100 + i}",
            "process_nm": rng.choice([7, 12, 16, 22, 28, 40, 55]),
            "cpu_arch": rng.choice(["cortex-m55", "cortex-m85", "cortex-a53",
                                    "cortex-a78", "riscv-rv32imc", "riscv-rv64gc"]),
            "cpu_mhz": rng.choice([80, 200, 400, 800, 1200, 1800, 2200]),
            "cores": rng.choice([1, 1, 2, 4, 4, 8]),
            "_vendor": vendor["id"],
        }
        socs.append(soc)

        # Every SoC has a CPU (the universal fallback), plus 0-2 accelerators.
        kinds = ["MCU-CPU"]
        extra = rng.choices([0, 1, 1, 2], k=1)[0]
        kinds += rng.sample([k[0] for k in ACCEL_ARCHETYPES if k[0] != "MCU-CPU"],
                            min(extra, 4))
        for kind in kinds:
            arch = next(a for a in ACCEL_ARCHETYPES if a[0] == kind)
            _, cats, opset_ceiling, gops_rng, sram_rng, energy = arch
            aid = _rid("accel", len(accelerators))
            accelerators.append({
                "id": aid,
                "name": f"{soc['name']}-{kind}",
                "kind": kind,
                "gops_int8": round(rng.uniform(*gops_rng), 2),
                "sram_kb": rng.choice([s for s in (64, 128, 256, 512, 1024, 2048, 4096,
                                                   8192, 16384, 32768)
                                       if sram_rng[0] <= s <= sram_rng[1]] or [sram_rng[0]]),
                "clock_mhz": rng.choice([100, 200, 400, 600, 800, 1000, 1400]),
                "opset_ceiling": opset_ceiling,
                "energy_factor": energy,
                # Whether running here *is* the CPU fallback, rather than
                # acceleration. Explicit because the catalog used to infer it
                # from `kind <> "MCU-CPU"`, which reads ONNX Runtime's CPU
                # execution provider (kind "CPU") as an accelerator (#69).
                "is_cpu_fallback": 1 if kind == "MCU-CPU" else 0,
                "_cats": cats, "_soc": soc["id"],
            })
    fleet.add_nodes("SoC", [{k: v for k, v in s.items() if not k.startswith("_")} for s in socs])
    fleet.add_nodes("Accelerator", [{k: v for k, v in a.items() if not k.startswith("_")}
                                    for a in accelerators])
    for s in socs:
        fleet.add_edge("SoC", s["id"], "MADE_BY", "Vendor", s["_vendor"], None)
    for a in accelerators:
        fleet.add_edge("SoC", a["_soc"], "HAS_ACCELERATOR", "Accelerator", a["id"], None)

    accel_by_soc: dict[str, list[dict]] = {}
    for a in accelerators:
        accel_by_soc.setdefault(a["_soc"], []).append(a)

    # ---------------- Runtime -> Accelerator ----------------
    for rt in runtimes:
        for a in accelerators:
            if a["kind"] in rt["_kinds"]:
                fleet.add_edge("Runtime", rt["id"], "TARGETS", "Accelerator", a["id"], None)

    # ---------------- Kernels ----------------
    # A Kernel is a concrete (operator, accelerator, runtime) implementation.
    # It exists only when the accelerator's archetype covers the operator's
    # category AND the operator's opset is within the accelerator's ceiling.
    # The gaps this leaves are the whole point of the KG.
    kernels = []
    for a in accelerators:
        rts = [r for r in runtimes if a["kind"] in r["_kinds"]]
        if not rts:
            continue
        chosen_rts = rng.sample(rts, min(len(rts), rng.choice([1, 2, 2, 3])))
        for rt in chosen_rts:
            for o in ops:
                if o.category not in a["_cats"]:
                    continue
                if o.since_version > a["opset_ceiling"]:
                    continue
                if o.is_control_flow and a["kind"] != "MCU-CPU":
                    continue
                # A few operators are simply missing from any given vendor's
                # kernel library -- the realistic, annoying case.
                if a["kind"] != "MCU-CPU" and rng.random() < 0.18:
                    continue
                kid = _rid("kernel", len(kernels))
                kernels.append({
                    "id": kid,
                    "name": f"{o.name}@{a['name']}/{rt['name']}",
                    "efficiency": round(rng.uniform(0.35, 0.98), 3),
                    "is_fallback": 1 if a["kind"] == "MCU-CPU" else 0,
                    "_op": o.id, "_accel": a["id"], "_rt": rt["id"],
                })
    fleet.add_nodes("Kernel", [{k: v for k, v in k2.items() if not k.startswith("_")}
                               for k2 in kernels])
    for k in kernels:
        fleet.add_edge("Kernel", k["id"], "IMPLEMENTS", "Operator", k["_op"], None)
        fleet.add_edge("Kernel", k["id"], "RUNS_ON", "Accelerator", k["_accel"], None)
        fleet.add_edge("Kernel", k["id"], "PROVIDED_BY", "Runtime", k["_rt"], None)

    # ---------------- Boards ----------------
    boards = []
    for i in range(n(120)):
        soc = rng.choice(socs)
        vendor = rng.choice(vendors)
        power = rng.choice([15, 30, 60, 120, 250, 500, 1200, 2500, 5000])
        boards.append({
            "id": _rid("board", i),
            "name": f"{vendor['name'].split()[0][:2].upper()}{rng.choice('XKNVR')}-{200 + i}",
            "form_factor": rng.choice(FORM_FACTORS),
            "price_usd": round(rng.uniform(9, 420), 2),
            "power_budget_mw": power,
            "ram_kb": rng.choice([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]),
            "flash_kb": rng.choice([256, 512, 1024, 2048, 4096, 8192, 16384, 65536]),
            "year": rng.choice([2022, 2023, 2023, 2024, 2024, 2025, 2026]),
            "battery_powered": 1 if power <= 250 else 0,
            "_soc": soc["id"], "_vendor": vendor["id"],
        })
    fleet.add_nodes("Board", [{k: v for k, v in b.items() if not k.startswith("_")}
                              for b in boards])
    for b in boards:
        fleet.add_edge("Board", b["id"], "HAS_SOC", "SoC", b["_soc"], None)
        fleet.add_edge("Board", b["id"], "MADE_BY", "Vendor", b["_vendor"], None)
        for c in rng.sample(certs, rng.choice([0, 1, 1, 2])):
            fleet.add_edge("Board", b["id"], "CERTIFIED_FOR", "Certification", c["id"], None)

    # ---------------- Clinical tasks / sensors / pipelines ----------------
    tasks = []
    for i, (nm, cat, mods, budget, sens) in enumerate(CLINICAL_TASKS):
        tid = _rid("task", i)
        tasks.append({"id": tid, "name": nm, "category": cat,
                      "latency_budget_ms": budget, "min_sensitivity": sens,
                      "_mods": mods})
        for s in sensors:
            if s["modality"] in mods:
                fleet.add_edge("ClinicalTask", tid, "REQUIRES_SENSOR", "Sensor", s["id"], None)
        for c in rng.sample(certs, rng.choice([1, 1, 2])):
            fleet.add_edge("ClinicalTask", tid, "GOVERNED_BY", "Certification", c["id"], None)
    fleet.add_nodes("ClinicalTask", [{k: v for k, v in t.items() if not k.startswith("_")}
                                     for t in tasks])

    # Sensor -> stage chains
    for s in sensors:
        chain = rng.sample(stages, rng.choice([3, 4, 4, 5]))
        fleet.add_edge("Sensor", s["id"], "FEEDS", "SignalStage", chain[0]["id"], None)
        for a, b in pairwise(chain):
            fleet.add_edge("SignalStage", a["id"], "NEXT_STAGE", "SignalStage", b["id"], None)
        s["_chain"] = chain

    # ---------------- Models + variants ----------------
    models, variants = [], []
    # Never generate fewer models than clinical tasks, so every task is
    # covered at any scale factor and task-anchored queries stay non-empty.
    model_total = max(n(60), len(tasks))
    for i in range(model_total):
        fam, cats, mac_rng = rng.choice(MODEL_FAMILIES)
        # Cover every clinical task at least once before assigning at random,
        # so task-anchored queries are never empty on a small scale factor.
        task = tasks[i] if i < len(tasks) else rng.choice(tasks)
        macs_m = round(rng.uniform(*mac_rng), 2)
        params_k = round(macs_m * rng.uniform(8, 60), 1)
        mid = _rid("model", i)
        models.append({"id": mid, "name": f"{fam}-{task['category']}-{i:03d}",
                       "family": fam, "task": task["name"], "params_k": params_k,
                       "macs_m": macs_m, "_cats": cats, "_task": task["id"]})

        # Operators this model uses, drawn from its family's categories.
        used = []
        for cat in cats:
            pool = ops_by_cat.get(cat, [])
            if not pool:
                continue
            used.extend(rng.sample(pool, min(len(pool), rng.choice([2, 3, 3, 4]))))
        for o in dict.fromkeys(used):
            fleet.add_edge("Model", mid, "USES_OPERATOR", "Operator", o.id,
                           {"count": rng.randint(1, 24)})
        models[-1]["_ops"] = list(dict.fromkeys(used))

        fleet.add_edge("Model", mid, "SOLVES", "ClinicalTask", task["id"], None)
        for d in rng.sample(datasets, rng.choice([1, 1, 2])):
            fleet.add_edge("Model", mid, "TRAINED_ON", "Dataset", d["id"], None)

        # Pipeline that feeds this model
        cand_sensors = [s for s in sensors if s["modality"] in task["_mods"]]
        if cand_sensors:
            src = rng.choice(cand_sensors)
            fleet.add_edge("SignalStage", src["_chain"][-1]["id"], "PRECEDES", "Model", mid, None)

        base_acc = rng.uniform(0.78, 0.985)
        fp32_size_kb = params_k * 4.0
        for pname, size_mult, thr_mult, acc_delta in PRECISIONS:
            vid = _rid("variant", len(variants))
            variants.append({
                "id": vid,
                "name": f"{models[-1]['name']}-{pname}",
                "precision": pname,
                "size_kb": round(fp32_size_kb * size_mult, 2),
                "accuracy": round(min(0.999, base_acc + acc_delta), 4),
                "format": rng.choice(["onnx", "tflite", "pte"]),
                "_model": mid, "_thr": thr_mult,
            })
            fleet.add_edge("ModelVariant", vid, "VARIANT_OF", "Model", mid, None)

    fleet.add_nodes("Model", [{k: v for k, v in m.items() if not k.startswith("_")}
                              for m in models])
    fleet.add_nodes("ModelVariant", [{k: v for k, v in v2.items() if not k.startswith("_")}
                                     for v2 in variants])

    # ---------------- Deployments ----------------
    # A Deployment is a measured (variant, board, runtime) triple. Metrics are
    # derived from the cost model in docs/data-provenance.md.
    models_by_id = {m["id"]: m for m in models}

    # kernel coverage lookup: (accel_id, runtime_id) -> set(op_id)
    coverage: dict[tuple[str, str], set[str]] = {}
    for k in kernels:
        coverage.setdefault((k["_accel"], k["_rt"]), set()).add(k["_op"])

    deployments = []
    per_variant = max(1, round(6 * scale))
    for v in variants:
        model = models_by_id[v["_model"]]
        for b in rng.sample(boards, min(len(boards), per_variant)):
            accels = accel_by_soc.get(b["_soc"], [])
            if not accels:
                continue
            # Pick the strongest accelerator on the board and a runtime that
            # can target it.
            accel = max(accels, key=lambda a: a["gops_int8"])
            rts = [r for r in runtimes if accel["kind"] in r["_kinds"]
                   and (accel["id"], r["id"]) in coverage]
            if not rts:
                continue
            rt = rng.choice(rts)
            cpu = next((a for a in accels if a["kind"] == "MCU-CPU"), accel)

            covered = coverage.get((accel["id"], rt["id"]), set())
            model_ops = model["_ops"]
            fallback_ops = [o for o in model_ops if o.id not in covered]
            frac_fb = len(fallback_ops) / max(1, len(model_ops))

            # Cost model: MACs split between accelerator and CPU fallback.
            macs = model["macs_m"] * 1e6
            thr = v["_thr"]
            acc_ops = accel["gops_int8"] * 1e9 * thr
            cpu_ops = cpu["gops_int8"] * 1e9 * thr
            t_acc = (macs * 2 * (1 - frac_fb)) / max(acc_ops, 1.0)
            t_cpu = (macs * 2 * frac_fb) / max(cpu_ops, 1.0)
            latency_ms = round((t_acc + t_cpu) * 1000
                               * rng.uniform(*LATENCY_JITTER), LATENCY_DECIMALS)

            energy_mj = ((t_acc * accel["energy_factor"] + t_cpu * cpu["energy_factor"])
                         * b["power_budget_mw"])
            power_mw = round(min(b["power_budget_mw"] * rng.uniform(0.35, 0.98),
                                 b["power_budget_mw"]), 2)
            memory_kb = round(v["size_kb"] * rng.uniform(1.15, 1.9), 2)
            fits = 1 if (memory_kb <= b["ram_kb"] and v["size_kb"] <= b["flash_kb"]) else 0

            did = _rid("deploy", len(deployments))
            # Placement draws from `site_rng`, **not** the shared `rng`. A
            # draw taken from the shared stream shifts every draw after it, so
            # one new field moves every deployment metric at an unchanged seed:
            # measured, `deploy:00001` reads 78.036 / 3591.96 that way against
            # 87.684 / 3990.6 here. This module's contract is that a seed
            # reproduces the graph byte-for-byte and
            # `docs/data-provenance.md`'s cost-model figures rest on it, so any
            # field added later needs its own stream for the same reason.
            #
            # One site per deployment: a deployment is one installed unit, so
            # it is in exactly one place -- which is what makes "how many of
            # this site's deployments are on the recalled board" a count
            # rather than a set union.
            site = site_rng.choice(sites)
            deployments.append({
                "id": did,
                "_site": site["id"],
                "latency_ms": latency_ms,
                "power_mw": power_mw,
                "energy_mj": round(energy_mj, 4),
                "memory_kb": memory_kb,
                "fallback_op_count": len(fallback_ops),
                "fallback_fraction": round(frac_fb, FALLBACK_FRACTION_DECIMALS),
                "accelerator_kind": accel["kind"],
                "fits": fits,
                "_variant": v["id"], "_board": b["id"], "_rt": rt["id"],
                "_accel": accel["id"],
            })
    fleet.add_nodes("Deployment", [{k: v2 for k, v2 in d.items() if not k.startswith("_")}
                                   for d in deployments])
    for d in deployments:
        fleet.add_edge("Deployment", d["id"], "OF_VARIANT", "ModelVariant", d["_variant"], None)
        fleet.add_edge("Deployment", d["id"], "ON_BOARD", "Board", d["_board"], None)
        fleet.add_edge("Deployment", d["id"], "VIA_RUNTIME", "Runtime", d["_rt"], None)
        fleet.add_edge("Deployment", d["id"], "USES_ACCELERATOR", "Accelerator", d["_accel"], None)
        fleet.add_edge("Deployment", d["id"], "DEPLOYED_AT", "Site", d["_site"], None)

    return fleet


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate the synthetic edge-AI fleet.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    f = generate(seed=args.seed, scale=args.scale)
    p = write(f)
    print(f"[generate] seed={f.seed} scale={f.scale} "
          f"nodes={f.node_count:,} edges={f.edge_count:,} -> {p}")
    for label, rows in sorted(f.nodes.items(), key=lambda kv: -len(kv[1])):
        print(f"           {label:16s} {len(rows):>7,}")

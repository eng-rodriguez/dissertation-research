"""EXP-017: node-strength FC rung (06 v1.1 §4.3 item 7; descriptive). Fits only B2 + node strength with the EXP-004
machinery (src/pipeline/confirmatory.run, unchanged) on the EXP-004 cache; the B2 and B2 + FC per-patient AUROCs are
the committed EXP-004 units (same manifest, folds and seeds), so the contrasts pair them within patient and repeat.

    python src/pipeline/exp017_node_strength.py run --workers 6
    python src/pipeline/exp017_node_strength.py summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in ("models", "evaluation", "features", "design", "pipeline", "visualization"):
    sys.path.insert(0, str(ROOT / "src" / p))
import confirmatory as cf  # noqa: E402
import exp010_epoch4s as e10  # noqa: E402
import run_ladder as rl  # noqa: E402

CONFIG = ROOT / "experiments/EXP-017-node-strength/config.json"
FROZEN_SHA256 = None  # set when Josue chooses to run it; run refuses until then
KIND = "hgb"


def load_config() -> tuple[dict, str]:
    raw = CONFIG.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def check_frozen(sha: str) -> None:
    if FROZEN_SHA256 is None or sha != FROZEN_SHA256:
        raise SystemExit(f"EXP-017 config is not the frozen one ({sha} vs {FROZEN_SHA256})")


def units(c: dict) -> list[dict]:
    """EXP-017 units plus the EXP-004 reference-model units (same 50 outer splits)."""
    import exp004_report as rep

    own = rep.load_units(KIND, ROOT / c["out"])
    ref = [json.loads(x) for x in (ROOT / c["reference_units"]).read_text().splitlines() if x.strip()]
    ref = [u for u in ref if u["model"] in c["reference_models"]]
    n = len(cf.design()["splits"]) * cf.f3.N_FOLDS
    if len(own) != n * len(c["models_run"]) or len(ref) != n * len(c["reference_models"]):
        raise SystemExit(f"EXP-017 incomplete: {len(own)} own units, {len(ref)} reference units")
    return own + ref


def summary(c: dict, sha: str, with_states: bool = True) -> dict:
    import exp004_report as rep

    auroc, folds = rep.per_repeat(units(c))
    res = {"experiment": "EXP-017", "learner": KIND, "config_sha256": sha,
           "status": "descriptive (06 §4.3 item 7, §4.4); specified after unblinding; never relabels RESULT-001",
           "contrasts": {f"{a}_vs_{b}": e10.contrast(auroc, folds, a, b) for a, b in c["contrasts"]},
           "mean_auroc": {m: e10.mean_auroc(auroc, m) for m in c["models_run"] + c["reference_models"]}}
    if with_states:
        import exp004_descriptive as dx

        L = dx.load(ROOT / c["cache"], ROOT / c["scores"], KIND, c["models_run"])
        ref = dx.load(ROOT / c["cache"], ROOT / "data/processed/exp004/scores", KIND, c["reference_models"])
        L["scores"].update(ref["scores"])
        base = {i: r["evaluable_states"] for i, r in L["recs"].items() if r.get("evaluable_states") and i in L["data"]}
        res["per_state"] = {st: {f"{a}_vs_{b}": dx.contrast(L, a, b, {i: [st] for i, s in base.items() if st in s},
                                                            labelled=False) for a, b in c["contrasts"]}
                            for st in dx.STATES}
        res["per_state"]["note"] = "estimate + 95% percentile bootstrap CI, no outcome label (R10); wake underpowered"
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    c, sha = load_config()
    check_frozen(sha)
    out = ROOT / c["out"]
    if a.command == "run":
        if (out / f"summary_{KIND}.json").exists():
            raise SystemExit("EXP-017 runs once: a summary already exists")
        if json.loads((ROOT / c["reference_cache_check"]).read_text())["result"] != "PASS":
            raise SystemExit("the EXP-004 cache check must have passed")
        z_star = json.loads((ROOT / c["z_star"]["file"]).read_text())["z_star"]
        if z_star != c["z_star"]["value"]:
            raise SystemExit("pilot z* differs from the frozen value")
        cf.run(ROOT / c["cache"], out, ROOT / c["scores"], c["models_run"], KIND, z_star, a.workers)
    else:
        res = summary(c, sha)
        res["provenance"] = rl.provenance()
        (out / f"summary_{KIND}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: res[k] for k in ("contrasts", "mean_auroc")}, indent=2))


if __name__ == "__main__":
    main()

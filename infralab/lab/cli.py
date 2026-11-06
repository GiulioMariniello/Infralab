"""CLI del lab: python -m infralab.lab {validate,run} spec.yaml

Esempi:
    python -m infralab.lab validate experiments/demo.yaml
    python -m infralab.lab run experiments/demo.yaml --workers 4
"""
import argparse
import logging
import sys
import time

from infralab.lab.io import write_outputs
from infralab.lab.provenance import collect_provenance
from infralab.lab.runner import run_experiment
from infralab.lab.spec import SpecError, load_spec, resolve_scenario


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="infralab.lab", description="Runner campagne InfraLab")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="valida una spec senza eseguirla")
    p_validate.add_argument("spec", help="path del file YAML/JSON della spec")

    p_run = sub.add_parser("run", help="esegue la campagna")
    p_run.add_argument("spec", help="path del file YAML/JSON della spec")
    p_run.add_argument("--workers", type=int, default=1, help="processi paralleli (default 1)")
    p_run.add_argument("--output-dir", default=None, help="override della cartella output")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        spec = load_spec(args.spec)
        scenario = resolve_scenario(spec.scenario)
    except (SpecError, OSError) as e:
        print(f"Spec non valida: {e}", file=sys.stderr)
        return 2

    if args.command == "validate":
        print(f"Spec '{spec.name}' valida: {spec.run_count} run "
              f"({len(spec.agents)} agenti x {len(spec.grid_points())} punti griglia x {spec.replicates} repliche), "
              f"scenario '{scenario.get('scenario_id', spec.scenario)}', orizzonte {spec.horizon_quarters} trimestri.")
        return 0

    started = time.time()
    with open(args.spec, "r", encoding="utf-8") as f:
        spec_text = f.read()

    def progress(done, total):
        print(f"\r[{done}/{total}] run completate", end="", flush=True)

    results = run_experiment(spec, workers=args.workers, scenario=scenario, progress_callback=progress)
    print()
    elapsed = time.time() - started

    manifest = {
        "spec_name": spec.name,
        "spec_path": args.spec,
        "scenario_id": scenario.get("scenario_id", spec.scenario),
        "run_count": spec.run_count,
        "completed": len(results["runs"]),
        "failed": len(results["errors"]),
        "elapsed_seconds": round(elapsed, 1),
        "seed": spec.seed,
        **collect_provenance(spec_text),
    }
    output_dir = args.output_dir or spec.output_dir
    written = write_outputs(output_dir, results, manifest, spec.output_formats)
    for label, path in written.items():
        print(f"  {label}: {path}")
    if results["errors"]:
        print(f"ATTENZIONE: {len(results['errors'])} run fallite (vedi errors.log)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

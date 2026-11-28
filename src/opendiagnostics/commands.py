"""The commands that work on finished runs: ``evaluate``, ``examples``, ``compare``, ``froc``,
``publish``, ``retire``, ``prepare``, and ``notebook``. ``fit`` is LightningCLI, in ``cli``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opendiagnostics.cli import _prepare, evaluate_run, examples_run


def _examples(args: argparse.Namespace) -> int:
    manifest = examples_run(args.run_dir, args.limit)
    print(f"wrote {len(manifest)} examples to {Path(args.run_dir) / 'examples'}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    results = evaluate_run(args.run_dir, args.split, args.threshold_from)
    print(json.dumps(results, indent=2, default=str))
    return 0


def _compare(args: argparse.Namespace) -> int:
    from opendiagnostics.metrics.compare import compare, read_predictions

    result = compare(
        read_predictions(args.a),
        read_predictions(args.b),
        metric=args.metric,
        replicates=args.replicates,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    floor = 1 / result.valid_replicates if result.valid_replicates else float("nan")
    p = f"< {floor:.4f}" if result.p_value == 0 else f"{result.p_value:.4f}"
    print(
        f"{result.metric}: A {result.a:.4f}, B {result.b:.4f}\n"
        f"B - A = {result.difference:+.4f} "
        f"[{round(result.confidence * 100)}% CI {result.lower:+.4f}, {result.upper:+.4f}], p {p}\n"
        f"{result.samples} samples from {result.patients} patients, "
        f"{result.valid_replicates} of {result.replicates} replicates valid"
    )
    return 0


def _froc(args: argparse.Namespace) -> int:
    from opendiagnostics.metrics.froc import pool, scan_metrics

    scans = pool(args.predictions)
    result = scan_metrics(scans, replicates=args.replicates, seed=args.seed)
    if args.json:
        print(json.dumps({k: v.to_dict() for k, v in result.items()}, indent=2))
        return 0
    for name, estimate in result.items():
        print(f"{name}: {estimate}")
    lesions = sum(len(s.truth) for s in scans)
    patients = len({s.patient_id for s in scans})
    print(f"{len(scans)} scans from {patients} patients, {lesions} lesions")
    return 0


def _publish(args: argparse.Namespace) -> int:
    from opendiagnostics.publish import publish_run

    print(
        publish_run(
            args.run_dir,
            args.project,
            include_predictions=not args.no_predictions,
            allow_dirty=args.allow_dirty,
            include_checkpoint=not args.no_checkpoint,
        )
    )
    return 0


def _retire(args: argparse.Namespace) -> int:
    from opendiagnostics.publish import retire

    run = retire(args.project)
    print(f"{args.project} is off the demo; it served {run}, which stays in the bucket")
    return 0


def _notebook(args: argparse.Namespace) -> int:
    from opendiagnostics.notebook import build_notebook

    projects = build_notebook(args.models, args.out)
    print(
        f"wrote {len(projects)} projects and {sum(len(p.runs) for p in projects)} runs to {args.out}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opendiagnostics", description="Train, evaluate, and publish OpenDiagnostics models."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "fit", help="train and evaluate one experiment (see `opendiagnostics fit --help`)"
    )

    evaluate = sub.add_parser("evaluate", help="re-evaluate a finished run")
    evaluate.add_argument("run_dir", type=Path)
    evaluate.add_argument("--split", action="append", help="split to evaluate (repeatable)")
    evaluate.add_argument(
        "--threshold-from",
        nargs="+",
        type=Path,
        metavar="FOLD_RUN",
        help="fit the binary threshold on these k-fold runs' out-of-fold predictions",
    )
    evaluate.set_defaults(handler=_evaluate)

    examples = sub.add_parser(
        "examples", help="render test images beside the model's attribution map or boxes"
    )
    examples.add_argument("run_dir", type=Path)
    examples.add_argument("--limit", type=int, default=8, help="most examples to render")
    examples.set_defaults(handler=_examples)

    compare = sub.add_parser(
        "compare", help="paired patient-level bootstrap of B minus A on the same test samples"
    )
    compare.add_argument("a", type=Path, help="predictions CSV of the reference model")
    compare.add_argument("b", type=Path, help="predictions CSV of the challenger")
    compare.add_argument("--metric", default="auroc", choices=["auroc", "auprc"])
    compare.add_argument("--replicates", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--json", action="store_true", help="print the result as JSON")
    compare.set_defaults(handler=_compare)

    froc = sub.add_parser(
        "froc",
        help="FROC and CPM over 3D detection predictions pooled from several runs, "
        "such as the ten LUNA16 test folds",
    )
    froc.add_argument("predictions", nargs="+", type=Path, help="predictions_test.csv files")
    froc.add_argument("--replicates", type=int, default=1000)
    froc.add_argument("--seed", type=int, default=0)
    froc.add_argument("--json", action="store_true", help="print the result as JSON")
    froc.set_defaults(handler=_froc)

    publish = sub.add_parser("publish", help="copy a finished run into a model project")
    publish.add_argument("run_dir", type=Path)
    publish.add_argument(
        "project", type=Path, help="model project, e.g. models/pneumoniamnist-convnext"
    )
    publish.add_argument(
        "--no-predictions",
        action="store_true",
        help="leave out per-sample predictions (when the dataset license forbids sharing them)",
    )
    publish.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="do not upload the checkpoint to the public weights bucket",
    )
    publish.add_argument("--allow-dirty", action="store_true", help="publish a dirty-tree run")
    publish.set_defaults(handler=_publish)

    retire = sub.add_parser(
        "retire", help="take a model project off the demo; its weights stay in the bucket"
    )
    retire.add_argument("project", help="project name in catalog.json, e.g. dermamnist-convnext")
    retire.set_defaults(handler=_retire)

    prepare = sub.add_parser("prepare", help="download and unpack a config's dataset, no training")
    prepare.add_argument("-c", "--config", required=True, type=Path)
    prepare.set_defaults(handler=_prepare)

    notebook = sub.add_parser("notebook", help="generate the lab notebook pages for the docs")
    notebook.add_argument("--models", type=Path, default=Path("models"))
    notebook.add_argument("--out", type=Path, default=Path("docs/notebook"))
    notebook.set_defaults(handler=_notebook)
    return parser

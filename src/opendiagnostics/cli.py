"""Command line.

``opendiagnostics fit -c CONFIG [--trainer.max_epochs=2 ...]`` trains with Lightning's
``LightningCLI``, then evaluates the selected checkpoint with bootstrap intervals. The other
commands, in ``commands``, work on finished runs and the demo catalog.

A run directory holds everything needed to audit or repeat an experiment::

    runs/<name>/<timestamp>-<commit>/
        config.yaml          the fully resolved config
        run.json             seed, git commit and dirty flag, package versions, hardware
        splits.json          sample IDs in each split, with patient and label counts
        metrics.csv          one row per epoch from Lightning's CSVLogger
        checkpoint.ckpt      weights of the epoch with the best validation monitor
        metrics.json         final metrics with bootstrap confidence intervals
        predictions_<split>.csv
        examples/            test images beside the model's attribution map or boxes
        train.log
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import numbers
import sys
import time
import warnings
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import yaml
from jsonargparse import ArgumentParser
from jsonargparse.typing import register_type
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers import CSVLogger

from opendiagnostics import provenance
from opendiagnostics.data import MedicalDataModule, summarize
from opendiagnostics.evaluation import EvalSettings, evaluate, write_json
from opendiagnostics.examples import write_examples
from opendiagnostics.modules import DiagnosticModule

log = logging.getLogger("opendiagnostics")

# run.json and splits.json are written before the first metric, so these always fire.
warnings.filterwarnings("ignore", message=r".*directory .* exists and is not empty")


def _number(value: object) -> int | float:
    if isinstance(value, str):
        for kind in (int, float):
            try:
                return kind(value)
            except ValueError:
                pass
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"expected a number, got {value!r}")
    return value


# torchvision annotates arguments such as RandomAffine's `degrees` as numbers.Number, which
# jsonargparse can parse but not write back, so the saved config.yaml would not load.
register_type(numbers.Number, serializer=lambda value: value, deserializer=_number)


def _key(value: object) -> str | int:
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise ValueError(f"expected a key name, got {value!r}")
    return value


# MONAI dictionary transforms annotate `keys` as Hashable, which jsonargparse cannot write
# back either.
register_type(Hashable, serializer=lambda value: value, deserializer=_key)


def _version(output_dir: str, name: str, commit: str | None) -> str:
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{(commit or 'nogit')[:8]}"
    version, suffix = stamp, 1
    while (Path(output_dir) / name / version).exists():
        version, suffix = f"{stamp}-{suffix}", suffix + 1
    return version


class OpenDiagnosticsCLI(LightningCLI):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("seed_everything_default", 0)
        kwargs.setdefault("save_config_kwargs", {"config_filename": "config.yaml"})
        defaults = {"deterministic": True, "num_sanity_val_steps": 0}
        kwargs["trainer_defaults"] = defaults | kwargs.get("trainer_defaults", {})
        super().__init__(
            DiagnosticModule,
            MedicalDataModule,
            *args,
            subclass_mode_model=True,
            subclass_mode_data=True,
            **kwargs,
        )

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        return {"fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"}}

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--name", type=str, required=True, help="run name; runs go to output_dir/name"
        )
        parser.add_argument("--output_dir", type=str, default="runs")
        parser.add_argument("--notes", type=str, default="")
        parser.add_class_arguments(EvalSettings, "eval")
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults(
            {
                "checkpoint.monitor": "val/auroc",
                "checkpoint.mode": "max",
                "checkpoint.filename": "checkpoint",
                "checkpoint.save_weights_only": True,
                "checkpoint.auto_insert_metric_name": False,
                "checkpoint.enable_version_counter": False,
            }
        )
        parser.link_arguments("data.info", "model.init_args.info", apply_on="instantiate")

    @property
    def settings(self) -> Any:
        return self.config[self.subcommand] if self.subcommand else self.config

    @property
    def checkpoint(self) -> ModelCheckpoint:
        callback = self.trainer.checkpoint_callback
        assert isinstance(callback, ModelCheckpoint)
        return callback

    def instantiate_trainer(self, **kwargs: Any) -> Any:
        s = self.config_init[self.subcommand] if self.subcommand else self.config_init
        self.git = provenance.git_info(Path.cwd())
        version = _version(s.output_dir, s.name, self.git.get("commit"))
        kwargs["logger"] = CSVLogger(save_dir=s.output_dir, name=s.name, version=version)
        return super().instantiate_trainer(**kwargs)

    def before_fit(self) -> None:
        self.started = provenance.utc_now()
        data = self.datamodule
        data.prepare_data()
        data.setup("fit")
        self.run_dir = Path(self.trainer.log_dir or ".")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint.dirpath = str(self.run_dir)
        self.handler = logging.FileHandler(self.run_dir / "train.log")
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        for name in ("opendiagnostics", "lightning.pytorch"):
            logging.getLogger(name).addHandler(self.handler)
        s = self.settings
        source, info = data.source, data.info
        self.record = {
            "name": s.name,
            "seed": s.seed_everything,
            "deterministic": s.trainer.deterministic,
            "git": self.git,
            "environment": provenance.environment(),
            "device": str(self.trainer.strategy.root_device),
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
            "dataset": {
                "name": type(data).__name__,
                "license": info.license,
                "citation": info.citation,
                "patient_ids_available": info.patient_ids_available,
                "fingerprint": source.fingerprint,
            },
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "started": self.started,
        }
        write_json(self.run_dir / "run.json", self.record)
        write_json(
            self.run_dir / "splits.json",
            {
                "method": data.split,
                "summary": summarize(data.splits, source.patient_ids, source.strata),
                "ids": {k: [source.sample_ids[i] for i in v] for k, v in data.splits.items()},
            },
        )
        log.info("run directory: %s", self.run_dir)

    def after_fit(self) -> None:
        try:
            checkpoint = Path(self.checkpoint.best_model_path)
            if not checkpoint.is_file():
                raise RuntimeError(
                    f"no checkpoint was saved; check that checkpoint.monitor "
                    f"{self.settings.checkpoint.monitor!r} is a logged validation metric"
                )
            s = self.settings
            results = evaluate(
                self.trainer,
                self.model,
                self.datamodule,
                EvalSettings(**vars(s.eval)),
                s.seed_everything,
                checkpoint,
                self.run_dir,
            )
            results["selection"] = {
                "monitor": s.checkpoint.monitor,
                "mode": s.checkpoint.mode,
                "best_epoch": results["checkpoint_epoch"],
            }
            write_json(self.run_dir / "metrics.json", results)
            write_examples(
                self.model,
                self.datamodule,
                self.run_dir,
                checkpoint,
                self.trainer.strategy.root_device,
            )
            self.record["finished"] = provenance.utc_now()
            write_json(self.run_dir / "run.json", self.record)
            log.info("finished: %s", self.run_dir / "metrics.json")
            print(self.run_dir)
        finally:
            for name in ("opendiagnostics", "lightning.pytorch"):
                logging.getLogger(name).removeHandler(self.handler)
            self.handler.close()


def fit(args: list[str] | None = None) -> Path:
    """Run ``fit`` with ``args`` (or the command line when ``None``); return the run directory."""
    cli = OpenDiagnosticsCLI(args=None if args is None else ["fit", *args])
    return cli.run_dir


def load_run(
    run_dir: Path, splits: list[str] | None = None, *, with_data: bool = True
) -> OpenDiagnosticsCLI:
    """Rebuild a finished run's data module, model, and trainer from its saved config.

    ``with_data=False`` skips downloading and splitting the data and does not fetch the
    network's pretrained weights, for callers that only need the model and its transforms,
    such as the inference service.
    """
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    skipped = OpenDiagnosticsCLI.subcommands()["fit"] | {"self"}
    for argument in set(inspect.signature(Trainer.fit).parameters) - skipped:
        config.pop(argument, None)
    if splits:
        config["eval"]["splits"] = splits
    if not with_data:
        network = config["model"]["init_args"]["network"]
        if "pretrained" in (network.get("init_args") or {}):
            network["init_args"]["pretrained"] = False
    cli = OpenDiagnosticsCLI(args=config, run=False, save_config_callback=None)
    if with_data:
        cli.datamodule.prepare_data()
        cli.datamodule.setup("fit")
    return cli


def load_data_module(config: dict[str, Any]) -> MedicalDataModule:
    """The config's data module alone, without building the model or trainer."""
    parser = ArgumentParser()
    parser.add_subclass_arguments(MedicalDataModule, "data")
    return parser.instantiate_classes(parser.parse_object({"data": config["data"]})).data


def _prepare(args: argparse.Namespace) -> int:
    config = yaml.safe_load(Path(args.config).read_text())
    load_data_module(config).prepare_data()
    print(f"prepared the data for {config.get('name', args.config)}")
    return 0


def evaluate_run(
    run_dir: Path | str,
    splits: list[str] | None = None,
    threshold_from: list[Path] | None = None,
) -> dict[str, Any]:
    """Re-score a finished run's checkpoint with the current code and rewrite metrics.json.

    The model selection is kept; ``rescored`` records the commit that produced the numbers.
    With ``threshold_from`` (k-fold runs), the binary threshold comes from their out-of-fold
    predictions instead of this run's validation split.
    """
    from opendiagnostics.crossval import out_of_fold_threshold

    run_dir = Path(run_dir)
    calibration = out_of_fold_threshold(run_dir, threshold_from) if threshold_from else None
    cli = load_run(run_dir, splits)
    results = evaluate(
        cli.trainer,
        cli.model,
        cli.datamodule,
        EvalSettings(**vars(cli.config.eval)),
        cli.config.seed_everything,
        run_dir / "checkpoint.ckpt",
        run_dir,
        calibration,
    )
    previous = json.loads((run_dir / "metrics.json").read_text())
    results["selection"] = previous["selection"]
    results["rescored"] = {"git": cli.git, "at": provenance.utc_now()}
    write_json(run_dir / "metrics.json", results)
    return results


def examples_run(run_dir: Path | str, limit: int = 8) -> list[dict[str, Any]]:
    """Render a finished run's example figures with its own checkpoint."""
    run_dir = Path(run_dir)
    cli = load_run(run_dir)
    return write_examples(
        cli.model,
        cli.datamodule,
        run_dir,
        run_dir / "checkpoint.ckpt",
        cli.trainer.strategy.root_device,
        limit,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr
    )
    command = sys.argv[1:] if argv is None else argv
    if command and command[0] == "fit":
        fit(None if argv is None else argv[1:])
        return 0
    from opendiagnostics.commands import build_parser

    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

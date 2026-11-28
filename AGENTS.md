# AGENTS.md

OpenDiagnostics is a monorepo of PyTorch models for medical image classification and
detection: open research models, not a medical device. The goal is models that beat
published results, with the statistics to back the claim.
Everything trains with Lightning through one `LightningCLI` in `src/opendiagnostics`. Use the
libraries (Lightning, timm, torchvision, MONAI, torchmetrics, medmnist, scikit-learn) instead
of writing your own; see "Deliberate hand-rolls" for the few exceptions. The docs site is the
lab notebook.
Read [docs/design.md](docs/design.md) before changing behavior; it holds the data shapes,
the evaluation rules, the hosting decision, and the roadmap. The dataset survey, with
licenses and access terms, is in [docs/datasets.md](docs/datasets.md).

## Commands

```bash
uv sync --locked --extra cpu --extra serve && uv run ruff format --check && uv run ruff check && uv run ty check && uv run pytest
```

That is the check command. CI runs it on Ubuntu with Python 3.12, then builds the docs and
smoke-trains every config in `configs/smoke/`. On an NVIDIA machine, use
`--extra cu132` instead of `--extra cpu`.

| Task | Command |
|---|---|
| One test file | `uv run pytest tests/test_fit.py` |
| Fill or update expectations | `uv run pytest --expect-create`, then `uv run xpt failures` and `uv run xpt accept <id>` |
| Train | `uv run opendiagnostics fit -c models/<name>/config.yaml [--trainer.max_epochs=2 ...]` |
| Download a config's dataset only | `uv run opendiagnostics prepare -c models/<name>/config.yaml` |
| Threshold from k-fold runs | `uv run opendiagnostics evaluate runs/<name>/<run> --threshold-from runs/<name>-fold0 ... runs/<name>-fold4` |
| Regenerate the MedMNIST recipes | `uv run python scripts/medmnist_recipes.py` |
| Compare two runs, paired | `uv run opendiagnostics compare <a>/predictions_test.csv <b>/predictions_test.csv` |
| Publish a run (uploads the checkpoint to the public weights bucket; see `docs/weights.md`) | `uv run opendiagnostics publish runs/<name>/<run-id> models/<name>` |
| Take a project off the demo (its weights stay in the bucket) | `uv run opendiagnostics retire <project>` |
| Docs | `uv run opendiagnostics notebook && uv run zensical serve` |

## Layout

| Path | Owns |
|---|---|
| `src/opendiagnostics/` | the pipeline: LightningCLI (`cli.py`), data modules (`data/`), Lightning modules (`modules/`), evaluation, metrics, publish, notebook |
| `models/<name>/` | one model on one benchmark: `config.yaml`, `MODEL_CARD.md`, `baselines.yaml`, `runs/<run-id>/` |
| `configs/smoke/` | seconds-long configs on generated data |
| `scripts/` | generators; `medmnist_recipes.py` writes the `models/*-convnext/config.yaml` files |
| `docs/` | design, dataset survey, site home page; `docs/notebook/` is generated and ignored |
| `deploy/`, `web/` | the demo site: inference API contract and data handling (`deploy/README.md`), Kubernetes manifests, SvelteKit frontend |
| `runs/` | local training output, ignored |

## Invariants

1. **Splits are patient-level.** No patient appears in two splits. `check_disjoint` runs
   in the data module's `setup()`, before training. Datasets without patient IDs use their official split, and the run
   records that leakage could not be checked.
2. **The test split is for final numbers.** Model selection, thresholds, and calibration
   use validation data only.
3. **Every headline metric has an interval.** A percentile bootstrap that resamples
   patients. A claim that one model beats another needs `opendiagnostics compare` (a paired
   test) or a published value outside our interval.
4. **Comparisons with published results use the benchmark's protocol.** Same split, same
   metric definition. Every number in `baselines.yaml` has a source.
5. **Published runs come from clean commits.** `opendiagnostics publish` refuses a dirty-tree run.
   Runs are deterministic on CPU with `trainer.deterministic: true`, and a test checks that.
6. **Respect dataset licenses.** Never commit or upload data, weights, or per-image
   predictions that a dataset's terms forbid sharing. The survey lists each dataset's terms.
7. **No patient data in git.** No DICOM headers, reports, or anything with PHI. Run files
   hold sample IDs and model outputs. The one exception is a published run's `examples/`:
   a few de-identified test images, copied by `opendiagnostics publish` only when the
   dataset's license is in `OPEN_LICENSES` (`publish.py`). Never widen that list to
   credentialed or unknown-license data.
8. **Config parsing is strict.** Unknown keys and wrong types are errors. LightningCLI
   (jsonargparse) enforces this; keep components typed so it can.
9. **Uploads to the demo site are never kept.** Uploaded images and their results are
   processed in memory for one request and never written to disk, stored, logged, or used
   for training. Request metadata (time, path, status, client IP, user agent) may be
   logged and kept for 14 days; it never includes uploaded content or file names. A change
   that keeps any uploaded data, logs request contents, adds analytics or a third-party
   script, or changes retention must update the privacy page (`web/src/routes/privacy`)
   and `deploy/README.md` in the same change, and needs the user's approval first.

## Stop and ask first

On top of the hl stop points, ask before:

- applying for, downloading, or accepting the terms of a registration or credentialed
  dataset (PhysioNet, Stanford AIMI, ADNI, and others);
- creating an external account, organization, or repository (Hugging Face, Zenodo), or
  uploading a checkpoint or results to one;
- publishing a release or turning on GitHub Pages.

Open datasets that need no account (MedMNIST, for example) can be downloaded without asking.

## How to make common changes

- **Add a dataset.** Subclass `MedicalDataModule` in `src/opendiagnostics/data/`, set
  `self.info` in `__init__`, and return a `Source` from `load()` with one patient ID per
  sample. Keep the dataset picklable and lazy (Windows workers pickle it). Test it on a tiny
  fixture written to `tmp_path`, never on real data.
- **Add a network or transform.** No code: name the class in the config by `class_path`.
  Use `dict_kwargs` for factories that take `**kwargs`, such as `timm.create_model`.
- **Add a task.** Subclass `DiagnosticModule` in `src/opendiagnostics/modules/`.
- **Add a metric.** Call `torchmetrics` from `src/opendiagnostics/metrics/`, returning NaN
  where the metric is undefined. Pin it to scikit-learn in a test.
- **Start a model project.** Copy `models/_template/` and follow
  [models/README.md](models/README.md). For MedMNIST, add the subset to
  `scripts/medmnist_recipes.py` and rerun it.

## Tests

- **Whole outputs use xpt.** For a test whose expected value is a whole output (rendered
  Markdown, CLI text, error messages, summaries), write `expect(actual)` from `xpt` and run
  `uv run pytest --expect-create` to fill in the value. Read the filled value before you
  commit: xpt records what the code did, not what it should do.
- **Intended changes go through xpt.** When an output changes on purpose, run
  `uv run xpt failures`, read each diff, then `uv run xpt accept --verify <id>` for the ones
  you mean. `--verify` reruns only the tests that hit those call sites. Don't run
  `xpt accept --all` on diffs you haven't read.
- **A bug is not accepted.** Fix the code. If you can't now, `uv run xpt reject <id>`.
  When accept reports `suspect`, the value looks like a failure (an exception repr, `None`
  or an empty list where there was data, a new `nan`). Use `--force` only when it is the
  intended result.
- **A `flaky:` line under a diff** means the path changed with no source change. Add the
  option it names (`ignore=`, `unordered=True`, `atol=`, or the `xpt_deterministic`
  fixture) instead of accepting the value.
- **Before you report done**, run `uv run xpt coverage -- pytest` and add tests for changed
  lines no test ran, or say why they need none. Use `uv run xpt show <id>` after
  `pytest --lf --xpt-trace` to see which changed code a failing test ran.
- **Rules use plain asserts.** Invariants and properties (a metric matches scikit-learn,
  patients never span splits, a run reproduces) use `assert`.
- **Tests stay hermetic.** No network except tests marked `network`, no real datasets, and
  no files outside `tmp_path`.

## Deliberate hand-rolls

- **The patient-level bootstrap and paired compare** (`metrics/bootstrap.py`,
  `metrics/compare.py`). No library resamples whole patients; `torchmetrics`'
  `BootStrapper` resamples images, which gives intervals that are too narrow. Each
  replicate calls `torchmetrics`.
- **LUNA16 FROC matching** (`metrics/froc.py`). It follows the official
  `noduleCADEvaluationLUNA16.py`: a hit is a detection center inside the lesion's inscribed
  ellipsoid, each lesion counts once at its best score, and detections on ignore boxes are
  dropped. MONAI's `compute_froc_curve_data` drops the operating point that keeps every
  detection, which changes CPM. The curve itself comes from scikit-learn.
- **MedMNIST unpacking** (`data/medmnist.py`). The `medmnist` package loads whole splits
  into memory; we stream each array out of the archive to a memory-mapped `.npy` file.

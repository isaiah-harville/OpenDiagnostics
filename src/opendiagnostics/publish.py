"""Copy a finished run into a model project so the lab notebook can show it.

A published run is ``models/<project>/runs/<run-id>/``. It holds the small files from the run
directory, which are committed to git and served by the docs site, and ``artifacts.yaml``,
which lists the large files (the checkpoint) by size, SHA-256, and download URL. When the
dataset's license allows sharing trained weights, the checkpoint is uploaded to the public
weights bucket (see ``docs/weights.md``) and the example figures (``examples/``) are copied;
otherwise neither leaves the machine.

The bucket is also what the demo serves. Publishing a run with its checkpoint uploads
``<project>/<run-id>/demo.tar`` (the run's config, metrics, artifacts, and examples) and
points the project's entry in ``catalog.json`` at that run, with the bundle's SHA-256 so the
demo notices when a rescored run's files change.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Protocol

import boto3
import yaml
from boto3.exceptions import S3UploadFailedError
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict

SMALL_FILES = ("config.yaml", "run.json", "metrics.json", "metrics.csv", "splits.json")
LARGE_FILES = ("checkpoint.ckpt",)
MAX_SMALL_FILE_BYTES = 20 * 1024 * 1024
EXAMPLES = "examples"
EXAMPLES_MANIFEST = "examples.json"
WEIGHTS_BUCKET = "opendiagnostics-weights"
WEIGHTS_URL = f"https://diagnostics.harville.dev/{WEIGHTS_BUCKET}"
# Cloudflare, in front of the weights host, refuses Python's default User-Agent.
USER_AGENT = "opendiagnostics"
# The homelab's S3 gateway has 256 MiB of memory and restarts under sustained uploads, which
# takes about 30 s. Upload one 16 MiB part at a time, and retry with backoff long enough to
# ride out a restart; retrying a part is safe.
UPLOAD_CONFIG = TransferConfig(max_concurrency=1, multipart_chunksize=16 * 1024 * 1024)
RETRIES = {"max_attempts": 10, "mode": "standard"}
# Licenses that allow redistributing a dataset's images and models trained on it, so example
# figures and checkpoints may be published. Anything else, including credentialed data and
# unknown licenses, publishes neither.
OPEN_LICENSES = frozenset(
    {"CC0 1.0", "CC BY 3.0", "CC BY 4.0", "CC BY-NC 4.0", "CC BY-NC-SA", "generated"}
)
CATALOG = "catalog.json"
DEMO_BUNDLE = "demo.tar"
DEMO_FILES = ("config.yaml", "metrics.json", "artifacts.yaml")


class PublishError(RuntimeError):
    """Raised when a run cannot be published as asked."""


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    bytes: int
    sha256: str
    url: str | None = None


class Artifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[Artifact]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifacts(path: Path) -> Artifacts:
    if not path.exists():
        return Artifacts(files=[])
    return Artifacts.model_validate(yaml.safe_load(path.read_text()) or {"files": []})


class Bucket(Protocol):
    def put(self, path: Path, key: str) -> None: ...

    def get(self, key: str) -> bytes | None: ...


class S3Bucket:
    """The weights bucket, written with the publisher key. Credentials and the endpoint come
    from the standard AWS variables; the client is made on first use."""

    def __init__(self) -> None:
        self._client = None

    def _s3(self):
        if self._client is None:
            self._client = boto3.client(
                "s3",
                config=Config(
                    s3={"addressing_style": "path"},
                    request_checksum_calculation="when_required",
                    retries=RETRIES,
                ),
            )
        return self._client

    def _failed(self, action: str, error: Exception) -> PublishError:
        return PublishError(
            f"{action} failed: {error}. Set AWS_ENDPOINT_URL_S3, AWS_ACCESS_KEY_ID, and "
            "AWS_SECRET_ACCESS_KEY to the publisher key (docs/weights.md), or pass "
            "--no-checkpoint to publish without weights."
        )

    def get(self, key: str) -> bytes | None:
        try:
            return self._s3().get_object(Bucket=WEIGHTS_BUCKET, Key=key)["Body"].read()
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise self._failed(f"Reading s3://{WEIGHTS_BUCKET}/{key}", error) from error
        except BotoCoreError as error:
            raise self._failed(f"Reading s3://{WEIGHTS_BUCKET}/{key}", error) from error

    def put(self, path: Path, key: str) -> None:
        """Upload ``path`` to ``key``. A checkpoint is then checked at its public URL, which
        is the one people download from."""
        try:
            self._s3().upload_file(
                str(path),
                WEIGHTS_BUCKET,
                key,
                ExtraArgs={"ContentType": "application/octet-stream"},
                Config=UPLOAD_CONFIG,
            )
        except (BotoCoreError, ClientError, S3UploadFailedError) as error:
            raise self._failed(
                f"Uploading {path.name} to s3://{WEIGHTS_BUCKET}/{key}", error
            ) from error
        if Path(key).name not in LARGE_FILES:
            return
        url = f"{WEIGHTS_URL}/{key}"
        try:
            request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=30) as r:
                served = int(r.headers.get("Content-Length", -1))
        except OSError as error:
            raise PublishError(
                f"Uploaded {path.name}, but {url} is not publicly readable: {error}. The bucket "
                "needs anonymous read and the public route on diagnostics.harville.dev."
            ) from error
        if served != path.stat().st_size:
            raise PublishError(f"{url} serves {served} bytes, expected {path.stat().st_size}")


def _publish_demo(bucket: Bucket, target: Path, project: str) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        bundle = Path(scratch) / DEMO_BUNDLE
        with tarfile.open(bundle, "w") as tar:
            for name in DEMO_FILES:
                tar.add(target / name, arcname=name)
            if (target / EXAMPLES).is_dir():
                tar.add(target / EXAMPLES, arcname=EXAMPLES)
        bucket.put(bundle, f"{project}/{target.name}/{DEMO_BUNDLE}")
        current = bucket.get(CATALOG)
        catalog = json.loads(current) if current else {"models": {}}
        catalog["models"][project] = target.name
        catalog["models"] = dict(sorted(catalog["models"].items()))
        bundles = catalog.setdefault("bundles", {})
        bundles[project] = sha256(bundle)
        catalog["bundles"] = dict(sorted(bundles.items()))
        listing = Path(scratch) / CATALOG
        listing.write_text(json.dumps(catalog, indent=2) + "\n")
        bucket.put(listing, CATALOG)


def retire(project: str, bucket: Bucket | None = None) -> str:
    """Take ``project`` off the demo by removing it from ``catalog.json``.

    Its checkpoints and bundles stay in the bucket, so publishing a run for it again restores it.
    Returns the run the demo served.
    """
    bucket = bucket or S3Bucket()
    current = bucket.get(CATALOG)
    catalog = json.loads(current) if current else {"models": {}}
    if project not in catalog["models"]:
        listed = ", ".join(catalog["models"]) or "none"
        raise PublishError(f"{project} is not in {CATALOG}; the demo serves: {listed}")
    run = catalog["models"].pop(project)
    catalog.get("bundles", {}).pop(project, None)
    with tempfile.TemporaryDirectory() as scratch:
        listing = Path(scratch) / CATALOG
        listing.write_text(json.dumps(catalog, indent=2) + "\n")
        bucket.put(listing, CATALOG)
    return run


def _license_allows_sharing(run_dir: Path) -> bool:
    record = json.loads((run_dir / "run.json").read_text())
    return record.get("dataset", {}).get("license") in OPEN_LICENSES


def publish_run(
    run_dir: Path,
    project_dir: Path,
    include_predictions: bool = True,
    allow_dirty: bool = False,
    include_checkpoint: bool = True,
    bucket: Bucket | None = None,
) -> Path:
    """Copy a finished run into ``project_dir/runs/<run-id>``.

    The small files are committed as they are. When the dataset's license allows sharing
    trained weights and ``include_checkpoint`` is true, the checkpoint is uploaded to
    ``<project>/<run-id>/checkpoint.ckpt`` in the weights bucket. ``artifacts.yaml`` records
    its size and SHA-256 either way, and its URL once uploaded. Rerunning on the same run
    keeps the recorded URL and does not upload the checkpoint again. Once the checkpoint is in
    the bucket, the run's demo bundle is uploaded and the catalog points the project at it.
    """
    bucket = bucket or S3Bucket()
    run_dir, project_dir = Path(run_dir), Path(project_dir)
    if not (run_dir / "metrics.json").exists():
        raise PublishError(f"{run_dir} has no metrics.json; publish a run that finished")
    if not (project_dir / "config.yaml").exists():
        raise PublishError(f"{project_dir} is not a model project (no config.yaml)")
    record = json.loads((run_dir / "run.json").read_text())
    if not record.get("finished"):
        raise PublishError(
            f"{run_dir} is still running: run.json has no finished time. Fit writes it after "
            "the example figures, so wait for the fit process to exit and publish again"
        )
    git = record.get("git", {})
    if git.get("dirty") and not allow_dirty:
        raise PublishError(
            f"{run_dir.name} ran on a dirty working tree, so its commit does not reproduce it. "
            "Rerun on a clean commit, or pass --allow-dirty to publish it anyway."
        )

    target = project_dir / "runs" / run_dir.name
    target.mkdir(parents=True, exist_ok=True)
    names = list(SMALL_FILES)
    if include_predictions:
        names += sorted(p.name for p in run_dir.glob("predictions_*.csv"))
    for name in names:
        source = run_dir / name
        if not source.exists():
            raise PublishError(f"{source} is missing")
        if source.stat().st_size > MAX_SMALL_FILE_BYTES:
            raise PublishError(f"{source} is over 20 MB; host it as a large artifact instead")
        shutil.copyfile(source, target / name)

    examples = run_dir / EXAMPLES
    shutil.rmtree(target / EXAMPLES, ignore_errors=True)
    if include_predictions and examples.is_dir() and _license_allows_sharing(run_dir):
        shutil.copytree(examples, target / EXAMPLES)

    known = {a.file: a for a in load_artifacts(target / "artifacts.yaml").files}
    artifacts = []
    for name in LARGE_FILES:
        source = run_dir / name
        if not source.exists():
            continue
        digest = sha256(source)
        previous = known.get(name)
        url = previous.url if previous is not None and previous.sha256 == digest else None
        if url is None and include_checkpoint and _license_allows_sharing(run_dir):
            key = f"{project_dir.name}/{run_dir.name}/{name}"
            bucket.put(source, key)
            url = f"{WEIGHTS_URL}/{key}"
        artifacts.append(Artifact(file=name, bytes=source.stat().st_size, sha256=digest, url=url))
    (target / "artifacts.yaml").write_text(
        yaml.safe_dump(Artifacts(files=artifacts).model_dump(), sort_keys=False)
    )
    if include_checkpoint and any(a.url for a in artifacts):
        _publish_demo(bucket, target, project_dir.name)
    return target

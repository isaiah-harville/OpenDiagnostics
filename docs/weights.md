# Model weights

Every published run whose dataset license allows sharing trained weights has a public,
downloadable checkpoint. Anyone can download one without an account.

## Download

- **Lab notebook.** Each run page has a **Download weights** button, with the file's SHA-256.
- **Demo site.** Each model page on [diagnostics.harville.dev](https://diagnostics.harville.dev)
  has the same button.
- **Directly.** Checkpoints are at

  ```
  https://diagnostics.harville.dev/opendiagnostics-weights/<project>/<run-id>/checkpoint.ckpt
  ```

  The URL, size, and SHA-256 of each one are in its run's `artifacts.yaml`, under
  `models/<project>/runs/<run-id>/`.

Check the file after downloading it:

```bash
curl -fLO https://diagnostics.harville.dev/opendiagnostics-weights/<project>/<run-id>/checkpoint.ckpt
shasum -a 256 checkpoint.ckpt
```

The hash must match `sha256` in `artifacts.yaml`.

## Load

A checkpoint is a Lightning checkpoint. To rebuild the model, you need the run's
`config.yaml` from the same folder as its `artifacts.yaml`. Put both in one folder:

```python
from pathlib import Path

import torch
from opendiagnostics.cli import load_run

folder = Path("pneumoniamnist-convnext")  # holds config.yaml and checkpoint.ckpt
model = load_run(folder, with_data=False).model
state = torch.load(folder / "checkpoint.ckpt", map_location="cpu", weights_only=True)
model.load_state_dict(state["state_dict"])
model.eval()
```

## License

The training data's license covers the weights. Weights are published only for datasets
whose license allows redistribution: CC0, CC BY, CC BY-NC, or generated data
(`OPEN_LICENSES` in `publish.py`). DermaMNIST is CC BY-NC 4.0, so its weights may be used
only non-commercially. Credentialed and unknown-license data never get public weights.

These are research models, not medical devices. They must not be used to diagnose or
treat anyone.

## Where they are stored

Checkpoints are in the public, read-only `opendiagnostics-weights` bucket on the homelab's
S3 storage. Git holds only each checkpoint's URL and SHA-256.

## Publish

`opendiagnostics publish` uploads the checkpoint when the dataset's license allows it. It
then confirms that the public URL serves the file and records the URL in `artifacts.yaml`.
It also uploads the run's demo files and points the project's entry in the bucket's
`catalog.json` at the run, so the demo site serves it within ten minutes.
It needs the publisher key and a machine on the LAN:

```bash
export AWS_ENDPOINT_URL_S3=https://s3.int.harville.dev
export AWS_ACCESS_KEY_ID=<publisher access key>
export AWS_SECRET_ACCESS_KEY=<publisher secret key>
uv run opendiagnostics publish runs/<name>/<run-id> models/<name>
```

The publisher key is the `opendiagnostics` identity in the homelab's
`apps/seaweedfs/seaweedfs/app/seaweedfs-s3-config.sops.yaml`. From a homelab checkout with
the age key, this prints it:

```bash
sops -d --extract '["stringData"]["seaweedfs_s3_config"]' \
    apps/seaweedfs/seaweedfs/app/seaweedfs-s3-config.sops.yaml \
  | jq '.identities[] | select(.name == "opendiagnostics") | .credentials[0]'
```

Publishing again does not upload an unchanged checkpoint. Publishing an older run makes the
demo serve that run again.
`--no-checkpoint` publishes the results without uploading the weights. Publish never
deletes a stored checkpoint. To remove one, delete the object with the publisher key
(`aws s3 rm s3://opendiagnostics-weights/<project>/<run-id>/checkpoint.ckpt` with the
variables above), then remove its `url` from `artifacts.yaml`. If the demo serves that run,
first retire the project or publish another run for it; the demo skips a cataloged run whose
checkpoint it cannot download.

## Retire

`opendiagnostics retire <project>` takes a project off the demo by removing it from
`catalog.json`, with the same publisher key as `publish`. Its checkpoints and bundles stay in
the bucket, and publishing a run for it again puts it back. The demo drops it at its next
catalog check, within ten minutes.

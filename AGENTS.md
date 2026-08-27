# get-tpu Agent Guide

## What this tool does

`get-tpu` manages Google Cloud TPU VM instances: create, start/stop, SSH config, disks, and cleanup. It wraps `gcloud` commands and maintains a local cache of known TPUs.

## How to invoke

Always use the shell wrapper — it handles the venv:

```bash
./get-tpu.sh <command> [args]
```

Set `VERBOSE=1` to see every `gcloud` command before it runs.

## Commands

For the available commands and their options, look at the `@app.command` decorators in `get-tpu.py` (or run `./get-tpu.sh --help`). `get-tpu.py` is the source of truth — this file does not list the commands.

## State files

| File | Purpose |
|------|---------|
| `~/.get-tpu/cache.json` | Tracks created TPUs; entries carry `type`, `zone`, and for flex-start also `queued_resource_id` and `kind` |
| `~/.get-tpu/config.json` | User config (created interactively on first run): `tpu_name_prefix`, `extra_startup_script`, `ssh_identity_file` |
| `~/.get-tpu/zones-cache.json` | Cached per-accelerator-type zone lists (`discover-zones`) |

Disks added with `add-disk` are recorded on the TPU's entry as a `disks`
list — `{"name": ..., "mount_point": ...}` — and are deleted together with
the TPU by `rm` and `flex-cleanup`.

`extra_startup_script` is called as `script STAGE_DIR`. It must not talk to the
TPU itself — it only writes local files into `STAGE_DIR`, which get packed into
the install payload and unpacked into `~` on the TPU. If it writes a `run.sh`
there, `run-all.sh` runs it after `setup.sh`. Anything under `STAGE_DIR/home/`
mirrors `$HOME` on the TPU.

## How an install runs

`create`, `reinstall`, and the flex paths all go through `install_tpu_script`, which:

1. waits for port 22, then for a usable SSH session
2. builds one payload tarball (`setup.sh`, `run-all.sh`, plus whatever the
   extra-startup hook staged) and copies it over with a single `scp`
3. unpacks it and runs `run-all.sh` **detached** under `setsid`/`nohup`, logging
   to `~/tpu-setup.log` on the TPU, and follows that log

Because the install is detached, losing the connection does not kill it. Re-run
`reinstall` to reattach to a running install, or read the log directly:

```bash
ssh <tpu-name> 'tail -f ~/tpu-setup.log'   # progress
ssh <tpu-name> 'cat ~/tpu-setup.log.rc'    # exit code, once finished
```

## TPU name convention

Names are `{tpu_name_prefix}{zone}` (flex-start appends `flex-` before the zone),
e.g. `tpu-vm-europe-west4-a`, `tpu-vm-flex-europe-west4-a`.

## SSH access

After `create`, `restart`, or a flex install, `~/.ssh/config` is updated
automatically with the TPU's external IP. Connect directly with:

```bash
ssh tpu-vm-europe-west4-a
```

## Prerequisites

- `gcloud` CLI authenticated and project set (`gcloud config get-value project`)
- `uv` (auto-installed by `get-tpu.sh` if missing)
- GCP project with TPU quota in target zones

## Supported zones

Europe first, then US, then Asia. Full list in `get-tpu.py:LOCATIONS`. Use
`discover-zones` to find which zones offer a given accelerator type.
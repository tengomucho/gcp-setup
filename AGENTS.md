# get-tpu Agent Guide

## What this tool does

`get-tpu` manages Google Cloud TPU VM instances: create, start/stop, SSH config, and cleanup. It wraps `gcloud` commands and maintains a local cache of known TPUs.

## How to invoke

Always use the shell wrapper — it handles the venv:

```bash
./get-tpu.sh <command> [args]
```

Set `VERBOSE=1` to see every `gcloud` command before it runs.

## Commands

| Command | Args | Description |
|---------|------|-------------|
| `create` | `[--accelerator-type TYPE] [--software-version VER] [--location ZONE]` | Create a new TPU VM, tries all locations until one succeeds |
| `restart` | `[NAME]` | Start a stopped TPU and update SSH config; if no name, tries all cached |
| `stop` | `[NAME]` | Stop a running TPU; if no name, stops the first running one found |
| `ls` | `[--details]` | List cached TPUs; `--details` fetches live state and IP |
| `rm` | `NAME` | Delete a TPU VM and remove from cache |
| `reinstall` | `NAME` | Re-run the setup script on an existing TPU |
| `print_config` | — | Show current config and cache file paths |
| `cleanup_ssh_hosts` | `[NAME]` | Remove stale known_hosts entries; if no name, cleans all cached |

## State files

| File | Purpose | Format |
|------|---------|--------|
| `~/.get-tpu/cache.json` | Tracks created TPUs | `{"tpu-name": {"type": "v5litepod-8", "zone": "europe-west4-a"}}` |
| `~/.get-tpu/config.json` | User config (optional) | JSON with fields below |

### Config fields (`~/.get-tpu/config.json`)

```json
{
  "tpu_name_prefix": "tpu-vm-",
  "extra_startup_script": "/path/to/script.sh",
  "ssh_identity_file": "~/.ssh/id_ed25519"
}
```

All fields are optional; defaults are used if the file is missing.

## TPU name convention

Names are `{tpu_name_prefix}{zone}`, e.g. `tpu-vm-europe-west4-a`.

## Common workflows

**Check what TPUs exist and their status:**
```bash
./get-tpu.sh ls --details
```

**Create a TPU (tries all zones automatically):**
```bash
./get-tpu.sh create --accelerator-type v5litepod-8 --software-version v2-alpha-tpuv5-lite
```

**Restart a stopped TPU (updates SSH config automatically):**
```bash
./get-tpu.sh restart tpu-vm-europe-west4-a
```

**Stop a running TPU to save cost:**
```bash
./get-tpu.sh stop tpu-vm-europe-west4-a
```

**Delete a TPU permanently:**
```bash
./get-tpu.sh rm tpu-vm-europe-west4-a
# Note: check and delete associated disks manually in GCP console
```

## SSH access

After `create` or `restart`, `~/.ssh/config` is updated automatically with the TPU's external IP. Connect directly with:

```bash
ssh tpu-vm-europe-west4-a
```

## Prerequisites

- `gcloud` CLI authenticated and project set (`gcloud config get-value project`)
- `uv` (auto-installed by `get-tpu.sh` if missing)
- GCP project with TPU quota in target zones

## Supported zones

Europe first, then US, then Asia. Full list in `get-tpu.py:LOCATIONS`. To target a specific zone, use `--location europe-west4-a`.

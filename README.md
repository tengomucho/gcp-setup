# GCP Setup

To setup, you can use the `get-tpu.sh` script (that will run a python script). use `--help` to get more info/help.
There are several subcommands, for now:

- create
- restart
- stop
- ls
- rm

## Handy commands

Sometimes it can be useful to know where a given type of TPU is available, e.g. for a v6e:

```bash
gcloud compute accelerator-types list \
  --filter="name=ct6e" \
  --format="value(zone)"
```

It is possible to filter by type, e.g. for v6e4:

```bash
gcloud compute machine-types list \
  --filter="name=ct6e-standard-4t" \
  --format="value(zone)"
```

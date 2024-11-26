import typer
import json
import os
from collections import OrderedDict
import subprocess
from rich import print


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = "cache.json"

# retrieved with gcloud compute tpus locations list --format=json
LOCATIONS = [
    "europe-west1-b",
    "europe-west1-c",
    "europe-west1-d",
    "europe-west4-a",
    "europe-west4-b",
    "europe-west4-c",
    "asia-east1-a",
    "asia-east1-b",
    "asia-east1-c",
    "asia-northeast1-b",
    "asia-southeast1-a",
    "asia-southeast1-b",
    "asia-southeast1-c",
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-central1-f",
    "us-east1-c",
    "us-east1-d",
    "us-east5-a",
    "us-east5-b",
    "us-east5-c",
    "us-south1-a",
    "us-south1-b",
    "us-south1-c",
    "us-west1-b",
    "us-west1-c",
    "us-west4-a",
    "us-west4-b"
]

app = typer.Typer()

def _run(cmd: str):
    split_cmd = cmd.split()
    subprocess.check_call(split_cmd)

def get_cache():
    cache_path = os.path.join(CUR_DIR, CACHE_FILE)
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def get_ext_ip(name: str, zone: str):
    output_with_ip = subprocess.getoutput(f"gcloud compute tpus tpu-vm describe --zone={zone} {name}")
    line_with_ip = [line for line in output_with_ip.split("\n") if "externalIp" in line][0]
    ext_ip = line_with_ip.split(":")[1].strip()
    return ext_ip

def get_state(name: str, zone: str):
    desc = subprocess.getoutput(f"gcloud compute tpus describe {name} --zone {zone}")
    lines = desc.split("\n")
    state = [line for line in lines if "state" in line][0].split(":")[1].strip()
    return state

def restart_tpu(name: str, zone: str):
    """Restart a TPU instance by name and zone.

    Args:
        name (str): Name of the TPU instance
        zone (str): Zone of the TPU instance
    """
    state = get_state(name, zone)
    if state == "READY":
        ext_ip = get_ext_ip(name, zone)
        print(f"🚀 TPU is ready at {ext_ip}, nothing to do.")
        return

    print(f"🚀 TPU [bold blue]{name}[/bold blue] is available, restarting..")
    _run(f"gcloud compute tpus tpu-vm start {name} --zone {zone}")
    print(f"TPU [bold blue]{name}[/bold blue] restarted, updating local IP/ssh records.")
    ext_ip = get_ext_ip(name, zone)
    print(f"External IP: {ext_ip}, updating ~/.ssh/config")
    with open(os.path.expanduser("~/.ssh/config"), "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if f"Host {name}" in line:
                lines[i+1] = f"  HostName {ext_ip}\n"
                break
    with open(os.path.expanduser("~/.ssh/config"), "w") as f:
        f.writelines(lines)
    print("Updating ~/.ssh/know_hosts")
    with open(os.path.expanduser("~/.ssh/known_hosts"), "r") as f:
        lines = f.readlines()
    # remove previous entry using the same ip
    lines = [line for line in lines if ext_ip not in line]
    # This is not great, because it bypasses ssh security check, but that's ok for these VMs
    new_entry = subprocess.getoutput(f"ssh-keyscan -H {ext_ip}")
    lines.append(new_entry)
    with open(os.path.expanduser("~/.ssh/known_hosts"), "w") as f:
        f.writelines(lines)
    print(f"✅ Done! You can now use [bold green]{name}[/bold green]")



@app.command()
def restart():
    print("[bold green]Getting TPU[bold green]")

    cache = get_cache()
    print(f"{len(cache)} elements in cache, trying to resume one of them...")
    for name in cache:
        instance = cache[name]
        zone = instance["zone"]
        print(f"\nChecking [bold blue]{name}[/bold blue] in [bold]{zone}[/bold]...")
        try:
            restart_tpu(name, zone)
            return
        except subprocess.CalledProcessError:
            print(f"❌ TPU [bold blue]{name}[/bold blue] is not available")
            continue


@app.command()
def stop():
    print("[bold green]Stopping TPU[bold green]")
    cache = get_cache()
    print(f"{len(cache)} elements in cache, trying to stop the first one that appears running.")
    for name in cache:
        instance = cache[name]
        zone = instance["zone"]
        print(f"\nChecking [bold blue]{name}[/bold blue] in [bold]{zone}[/bold]...")
        state = get_state(name, zone)
        if state == "READY":
            print(f"Stopping TPU [bold blue]{name}[/bold blue] in [bold]{zone}[/bold]...")
            _run(f"gcloud compute tpus tpu-vm stop {name} --zone {zone}")
            print(f"🧘 TPU [bold blue]{name}[/bold blue] stopped")
            return
        else:
            print(f"TPU {name} is not running, (state: [cyan]{state}[/cyan]) skipping..")


if __name__ == "__main__":
    app()

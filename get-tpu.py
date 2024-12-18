from dataclasses import dataclass
import typer
import json
import os
from collections import OrderedDict
import subprocess
import shlex
import getpass
from rich import print
from rich.table import Table
from rich.console import Console


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = "cache.json"
CONFIG_FILE = "~/.get-tpu-config.json"

# retrieved with gcloud compute tpus locations list --format=json
# manually resorted to to have europe first, then us, then asia
LOCATIONS = [
    "europe-west1-b",
    "europe-west1-c",
    "europe-west1-d",
    "europe-west4-a",
    "europe-west4-b",
    "europe-west4-c",
    "us-west1-b",
    "us-west1-c",
    "us-west4-a",
    "us-west4-b",
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
    "asia-east1-a",
    "asia-east1-b",
    "asia-east1-c",
    "asia-northeast1-b",
    "asia-southeast1-a",
    "asia-southeast1-b",
    "asia-southeast1-c",
]

app = typer.Typer()


@dataclass
class Config:
    tpu_name_prefix: str = "tpu-vm-"
    extra_startup_script: str = None
    ssh_identity_file: str = None


def _run(cmd: str):
    split_cmd = shlex.split(cmd)
    subprocess.check_call(split_cmd)


def get_cache():
    cache_path = os.path.join(CUR_DIR, CACHE_FILE)
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def get_config():
    config = Config()
    config_path = os.path.expanduser(CONFIG_FILE)
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
            for key in data:
                setattr(config, key, data[key])
    except FileNotFoundError:
        print(f"Config file not found at {config_path}, using default values.")
    return config


def get_project():
    value = subprocess.getoutput("gcloud config get-value project --format=json")
    value = value.replace('"', "")
    return value


def get_ext_ip(name: str, zone: str):
    output_with_ip = subprocess.getoutput(
        f"gcloud compute tpus tpu-vm describe --zone={zone} {name}"
    )
    line_with_ip = [
        line for line in output_with_ip.split("\n") if "externalIp" in line
    ][0]
    ext_ip = line_with_ip.split(":")[1].strip()
    return ext_ip


def get_state(name: str, zone: str):
    desc = subprocess.getoutput(f"gcloud compute tpus describe {name} --zone {zone}")
    lines = desc.split("\n")
    state = [line for line in lines if "state" in line][0].split(":")[1].strip()
    return state


def update_ssh_config(name: str, zone: str):
    print(
        f"TPU [bold blue]{name}[/bold blue] restarted, updating local IP/ssh records."
    )
    ext_ip = get_ext_ip(name, zone)
    print(f"External IP: {ext_ip}, updating ~/.ssh/config")
    with open(os.path.expanduser("~/.ssh/config"), "r") as f:
        host_found = False
        lines = f.readlines()
        for i, line in enumerate(lines):
            if f"Host {name}" in line:
                lines[i + 1] = f"  HostName {ext_ip}\n"
                host_found = True
                break
        if not host_found:
            lines.append(f"Host {name}\n")
            lines.append(f"  HostName {ext_ip}\n")
            current_user = getpass.getuser()
            lines.append(f"  User {current_user}\n")
            config = get_config()
            if config.ssh_identity_file:
                lines.append(f"  IdentityFile {config.ssh_identity_file}\n")
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
    update_ssh_config(name, zone)
    print(f"✅ Done! You can now use [bold green]{name}[/bold green]")


@app.command()
def create(
    accelerator_type: str = "v5litepod-8",
    software_version: str = "v2-alpha-tpuv5-lite",
    location: str = None,
):
    print("[bold green]Creating TPU[bold green]")
    cache = get_cache()
    if cache:
        print(
            f"⚠️ {len(cache)} elements in cache, It might be worth trying to resume one of them."
        )

    config = get_config()
    project = get_project()
    if location:
        locations = [location]
    else:
        locations = LOCATIONS
    for location in locations:
        print(f"\nTrying to create a TPU VM in [bold]{location}[/bold]...")
        name = f"{config.tpu_name_prefix}{location}"
        try:
            print("First check if the TPU is already created...")
            command = f"gcloud compute tpus tpu-vm describe {name} --zone {location}"
            out = subprocess.check_output(command, shell=True)
            print(f"🚀 TPU already exists in [bold]{location}[/bold] stopping script")
            return
        except subprocess.CalledProcessError:
            print("TPU not found, creating...")
        try:
            command = f"gcloud alpha compute tpus tpu-vm create {name} --zone {location} --accelerator-type={accelerator_type} --version={software_version}"
            _run(command)
            print(f"🚀 TPU created in [bold]{location}[/bold]")
            print(
                f"Updating cache with [bold blue]{name}[/bold blue] in [bold]{location}[/bold]..."
            )
            cache[name] = {"type": accelerator_type, "zone": location}
            with open(os.path.join(CUR_DIR, CACHE_FILE), "w") as f:
                json.dump(cache, f, indent=2)
            print("🧾 Copying setup script")
            _run(
                f"gcloud compute tpus tpu-vm scp --zone {location} setup.sh {name}: --project {project}"
            )
            print("🤖 Retrieving IP and updating local ssh settings")
            update_ssh_config(name, location)
            print("🏃 Running install script")
            _run(
                f"gcloud compute tpus tpu-vm ssh --zone {location} {name} --project {project} --command='bash setup.sh'"
            )
            print()
            if config.extra_startup_script:
                print(f"🔧 Running extra startup script {config.extra_startup_script}")
                subprocess.check_call(
                    f"{config.extra_startup_script} {name} {location}", shell=True
                )

            print(f"✅ Done! You can now use [bold green]{name}[/bold green]")
            return
        except subprocess.CalledProcessError:
            print(
                f"❌ TPU not available in [bold]{location}[/bold]"
            )
            continue


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
    print(
        f"{len(cache)} elements in cache, trying to stop the first one that appears running."
    )
    for name in cache:
        instance = cache[name]
        zone = instance["zone"]
        print(f"\nChecking [bold blue]{name}[/bold blue] in [bold]{zone}[/bold]...")
        state = get_state(name, zone)
        if state == "READY":
            print(
                f"Stopping TPU [bold blue]{name}[/bold blue] in [bold]{zone}[/bold]..."
            )
            _run(f"gcloud compute tpus tpu-vm stop {name} --zone {zone}")
            print(f"🧘 TPU [bold blue]{name}[/bold blue] stopped")
            return
        else:
            print(
                f"TPU {name} is not running, (state: [cyan]{state}[/cyan]) skipping.."
            )


@app.command()
def cache(check_state: bool = False):
    print("[bold green]Listing cached TPUs[bold green]")
    cache = get_cache()
    if check_state:
        table = Table("Name", "Zone", "State")
    else:
        table = Table("Name", "Zone")
    for name in cache:
        instance = cache[name]
        zone = instance["zone"]
        state_str = ""
        if check_state:
            state = get_state(name, zone)
            table.add_row(name, zone, state)
        else:
            table.add_row(name, zone)
    Console().print(table)


@app.command()
def delete(name: str):
    print(f"[bold green]Deleting TPU {name}[bold green]")
    cache = get_cache()
    if name not in cache:
        print(f"❌ TPU {name} not found in cache, delete it manually.")
        return
    instance = cache[name]
    zone = instance["zone"]
    print(f"Deleting TPU [bold blue]{name}[/bold blue] in [bold]{zone}[/bold]...")
    try:
        _run(f"gcloud compute tpus tpu-vm delete {name} --zone {zone}")
    except subprocess.CalledProcessError:
        print(f"❌ TPU {name} could not be deleted.")
        return
    del cache[name]
    with open(os.path.join(CUR_DIR, CACHE_FILE), "w") as f:
        json.dump(cache, f, indent=2)
    print(f"✅ TPU [bold blue]{name}[/bold blue] deleted")
    print(f"[bold orange]Note:[/bold orange] check if disks need to be deleted too.")


if __name__ == "__main__":
    app()

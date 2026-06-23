import getpass
import json
import os
import shlex
import shutil
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

import typer
from rich import print
from rich.console import Console
from rich.table import Table

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.expanduser("~/.get-tpu")
CACHE_FILE = os.path.join(CONFIG_DIR, "cache.json")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
VERBOSE = os.getenv("VERBOSE", "0") == "1"

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
    "us-east4-a",
    "us-east4-b",
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
    extra_startup_script: str | None = None
    ssh_identity_file: str | None = None


def _run(cmd: str):
    if VERBOSE:
        print(f"[bold blue]Running command:[/bold blue] {cmd}")
    split_cmd = shlex.split(cmd)
    subprocess.check_call(split_cmd)


def get_cache():
    cache_path = CACHE_FILE
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def _create_config_interactively() -> Config:
    config = Config()
    username = getpass.getuser()
    suggested_prefix = f"{username}-tpu-dev-"

    print(f"\nNo config file found at [bold]{CONFIG_FILE}[/bold]. Let's create one.")
    print()

    ok = typer.confirm(
        f"You can define a prefix for your TPU instances. Suggested: '{suggested_prefix}'. Is it ok?",
        default=True,
    )
    if ok:
        config.tpu_name_prefix = suggested_prefix
    else:
        config.tpu_name_prefix = typer.prompt(
            "Which prefix do you want?", default=suggested_prefix
        )

    script_path = typer.prompt(
        "\nYou can define a path to a script invoked after creating the TPU "
        "(called with tpu_name and zone as args). Press return to leave empty",
        default="",
    )
    config.extra_startup_script = script_path if script_path else None

    identity_file = typer.prompt(
        "\nIndicate the path of the SSH identity file you want to use. Press return to leave empty",
        default="",
    )
    config.ssh_identity_file = identity_file if identity_file else None

    if not os.access(CONFIG_DIR, os.F_OK):
        os.makedirs(CONFIG_DIR)
    with open(CONFIG_FILE, "w") as f:
        json.dump(
            {
                "tpu_name_prefix": config.tpu_name_prefix,
                "extra_startup_script": config.extra_startup_script,
                "ssh_identity_file": config.ssh_identity_file,
            },
            f,
            indent=2,
        )
    print(f"\n[bold green]Config saved to {CONFIG_FILE}[/bold green]")
    return config


def get_config():
    config = Config()
    config_path = CONFIG_FILE
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
            for key in data:
                setattr(config, key, data[key])
    except FileNotFoundError:
        config = _create_config_interactively()
    return config


def get_project():
    value = subprocess.getoutput("gcloud config get-value project --format=json")
    value = value.replace('"', "")
    return value


def list_tpus(zone: str):
    desc = subprocess.getoutput(
        f"gcloud compute tpus tpu-vm list --zone {zone} --format json"
    )
    # convert to json
    desc = json.loads(desc)
    return desc


def get_ext_ip(name: str, zone: str):
    desc = list_tpus(zone)
    filtered_desc = [item for item in desc if item["name"].endswith(name)]
    cur_tpu = filtered_desc[0]
    external_ip = cur_tpu["networkEndpoints"][0]["accessConfig"]["externalIp"]  # type: ignore
    return external_ip


def get_state(name: str, zone: str):
    desc = list_tpus(zone)
    filtered_desc = [item for item in desc if item["name"].endswith(name)]
    if not filtered_desc:
        return "NOT FOUND"
    state = filtered_desc[0]["state"]
    return state


def describe_queued_resource(queued_resource_id: str, zone: str) -> dict:
    out = subprocess.getoutput(
        f"gcloud alpha compute tpus queued-resources describe"
        f" {queued_resource_id} --zone {zone} --format json"
    )
    return json.loads(out)


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
    # Finally, cleanup known_hosts.
    cleanup_known_hosts(name)


def cleanup_known_hosts(ssh_alias: str):
    """Remove all known_hosts entries matching the host keys of the given SSH alias.

    This function:
    1. Resolves the SSH alias to get hostname and port
    2. Fetches all host keys from the server
    3. Removes all known_hosts entries with matching keys

    Args:
        ssh_alias (str): SSH alias or hostname to clean up
    """
    print(f"Resolving SSH configuration for '{ssh_alias}'...")

    # Use ssh -G to get the resolved configuration
    try:
        ssh_config = subprocess.getoutput(f"ssh -G {shlex.quote(ssh_alias)}")
        host = None
        port = None
        for line in ssh_config.split("\n"):
            if line.startswith("hostname "):
                host = line.split()[1]
            elif line.startswith("port "):
                port = line.split()[1]

        if not host:
            print(
                f"[bold red]Error:[/bold red] Could not resolve hostname for '{ssh_alias}'"
            )
            return

        print(f"Resolved to: {host}:{port}")
        print()

    except Exception as e:
        print(f"[bold red]Error:[/bold red] Failed to resolve SSH configuration: {e}")
        return

    # Backup known_hosts
    known_hosts = os.path.expanduser("~/.ssh/known_hosts")
    if not os.path.exists(known_hosts):
        print(f"No known_hosts file found at {known_hosts}")
        return

    backup_path = f"{known_hosts}.backup"
    shutil.copy2(known_hosts, backup_path)
    print(f"Backed up known_hosts to {backup_path}")
    print()

    # Fetch host keys from the server
    print(f"Fetching host keys from {host}:{port}...")
    try:
        keyscan_cmd = f"ssh-keyscan -p {port} -t rsa,ecdsa,ed25519 {shlex.quote(host)} 2>/dev/null"
        host_keys_output = subprocess.getoutput(keyscan_cmd)

        if not host_keys_output or host_keys_output.strip() == "":
            return

        # Extract all key parts (third field from each line)
        keys_to_remove = []
        for line in host_keys_output.split("\n"):
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 3:
                    keys_to_remove.append(parts[2])

        if not keys_to_remove:
            return

    except Exception:
        return

    # Read known_hosts and find matching entries
    try:
        with open(known_hosts, "r") as f:
            lines = f.readlines()

        # Find all matching entries
        matching_hosts = set()
        matching_lines = []

        for line in lines:
            for key in keys_to_remove:
                if key in line:
                    matching_lines.append(line)
                    # Extract hostname/IP (first field)
                    hostname = line.split()[0] if line.split() else ""
                    if hostname:
                        matching_hosts.add(hostname)
                    break

        if not matching_lines:
            print("No entries found with these host keys")
            return

        # Show unique hostnames/IPs that will be removed
        print("Entries that will be removed:")
        for host in sorted(matching_hosts):
            print(f"  {host}")

        print()
        print(f"Total entries to remove: {len(matching_lines)}")

        # Remove all entries matching any of these keys
        filtered_lines = []
        for line in lines:
            should_keep = True
            for key in keys_to_remove:
                if key in line:
                    should_keep = False
                    break
            if should_keep:
                filtered_lines.append(line)

        # Write back to known_hosts
        with open(known_hosts, "w") as f:
            f.writelines(filtered_lines)

        print("[bold green]✅ Successfully cleaned up known_hosts[/bold green]")

    except Exception as e:
        print(f"[bold red]Error:[/bold red] Failed to update known_hosts: {e}")
        # Restore backup
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, known_hosts)
            print("Restored known_hosts from backup")


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

    print(
        f"🚀 TPU [bold blue]{name}[/bold blue] is available, restarting at {datetime.now().isoformat()}..."
    )
    start_time = time.time()
    _run(f"gcloud compute tpus tpu-vm start {name} --zone {zone}")
    update_ssh_config(name, zone)
    print(
        f"✅ Done! Restarted [bold green]{name}[/bold green] in {time.time() - start_time} seconds"
    )


def install_tpu_script(name: str, location: str, project: str, config: Config):
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


@app.command()
def reinstall(name: str):
    """Re-run the setup script on an existing TPU VM."""
    cache = get_cache()
    if name not in cache:
        raise ValueError(f"❌ TPU {name} not found in cache, cannot reinstall it.")
    instance = cache[name]
    location = instance["zone"]
    project = get_project()
    install_tpu_script(name, location, project, get_config())


@app.command()
def create(
    accelerator_type: str = "v6e-4",
    software_version: str = "v2-alpha-tpuv6e",
    location: str | None = None,
):
    """Create a new TPU VM, trying all zones until one succeeds."""
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
        print("First check if the TPU is already created...")
        desc = list_tpus(location)
        if len(desc) > 0:
            print(
                f"🚀 TPU already exists in [bold]{location}[/bold], skipping this location."
            )
            continue

        print(f"TPU not found, creating at {datetime.now().isoformat()}...")
        start_time = time.time()
        try:
            command = f"gcloud alpha compute tpus tpu-vm create {name} --zone {location} --accelerator-type={accelerator_type} --version={software_version}"
            _run(command)
            print(
                f"🚀 TPU created in [bold]{location}[/bold] in {time.time() - start_time} seconds"
            )
            print(
                f"Updating cache with [bold blue]{name}[/bold blue] in [bold]{location}[/bold]..."
            )
            cache[name] = {"type": accelerator_type, "zone": location}
            if not os.access(CONFIG_DIR, os.F_OK):
                os.makedirs(CONFIG_DIR)
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f, indent=2)
            install_tpu_script(name, location, project, config)
            return
        except subprocess.CalledProcessError:
            print(f"❌ TPU not available in [bold]{location}[/bold]")
            continue


@app.command()
def restart(name: str | None = None):
    """Start a stopped TPU and update SSH config. If no name, tries all cached TPUs."""
    cache = get_cache()
    print("[bold green]Restarting TPU[bold green]")
    if name:
        if name not in cache:
            print(f"❌ TPU {name} not found in cache, cannot stop it.")
            return -1
        print(f"Restarting TPU [bold blue]{name}[/bold blue]...")
        cache = {name: cache[name]}
    else:
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
def stop(name: str | None = None):
    """Stop a running TPU to save cost. If no name, stops the first running one found."""
    cache = get_cache()
    if name:
        if name not in cache:
            print(f"❌ TPU {name} not found in cache, cannot stop it.")
            return -1
        print(f"Stopping TPU [bold blue]{name}[/bold blue]...")
        cache = {name: cache[name]}
    else:
        print("[bold green]Stopping TPU[bold green]")
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
def ls(details: bool = False):
    """List cached TPUs. Use --details to fetch live state and IP from GCP."""
    print("[bold green]Listing cached TPUs[bold green]")
    cache = get_cache()
    if details:
        table = Table("Name", "Zone", "Type", "State", "IP")
    else:
        table = Table("Name", "Zone")
    for name in cache:
        instance = cache[name]
        zone = instance["zone"]
        if details:
            state = get_state(name, zone)
            if state == "READY":
                ip = get_ext_ip(name, zone)
            elif state == "NOT FOUND":
                ip = "N/A"
            else:
                ip = ""
            tpu_type = instance["type"]
            table.add_row(name, zone, tpu_type, state, ip)
        else:
            table.add_row(name, zone)
    Console().print(table)


@app.command()
def rm(name: str):
    """Delete a TPU VM and remove it from cache."""
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
    if not os.access(CONFIG_DIR, os.F_OK):
        os.makedirs(CONFIG_DIR)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"✅ TPU [bold blue]{name}[/bold blue] deleted")
    print("[bold orange]Note:[/bold orange] check if disks need to be deleted too.")


@app.command()
def flex_start(
    zone: str,
    accelerator_type: str = "v6e-4",
    software_version: str = "v2-alpha-tpuv6e",
    max_run_duration: str = "9h",
):
    """Submit a flex-start (spot-like) queued resource request for a TPU."""
    config = get_config()
    cache = get_cache()
    node_id = f"{config.tpu_name_prefix}flex-{zone}"
    queued_resource_id = node_id

    print(f"[bold green]Submitting flex-start request[/bold green]")
    print(f"  Node ID:            [bold blue]{node_id}[/bold blue]")
    print(f"  Zone:               [bold]{zone}[/bold]")
    print(f"  Accelerator type:   {accelerator_type}")
    print(f"  Runtime version:    {software_version}")
    print(f"  Max run duration:   {max_run_duration}")

    command = (
        f"gcloud alpha compute tpus queued-resources create {queued_resource_id}"
        f" --zone={zone}"
        f" --accelerator-type={accelerator_type}"
        f" --runtime-version={software_version}"
        f" --node-id={node_id}"
        f" --provisioning-model=flex-start"
        f" --max-run-duration={max_run_duration}"
    )
    try:
        _run(command)
    except subprocess.CalledProcessError:
        print(f"❌ Failed to submit flex-start request for [bold]{zone}[/bold]")
        return

    cache[node_id] = {
        "type": accelerator_type,
        "zone": zone,
        "queued_resource_id": queued_resource_id,
        "kind": "flex-start",
    }
    if not os.access(CONFIG_DIR, os.F_OK):
        os.makedirs(CONFIG_DIR)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    print(
        f"\n✅ Queued resource [bold blue]{queued_resource_id}[/bold blue] submitted."
        f" Use [bold]flex-status[/bold] to monitor its state."
    )


_STATE_COLORS = {
    "ACTIVE": "bold green",
    "WAITING_FOR_RESOURCES": "yellow",
    "PROVISIONING": "yellow",
    "FAILED": "bold red",
    "SUSPENDING": "red",
    "SUSPENDED": "red",
}


@app.command()
def flex_status(name: str | None = None):
    """Show the status of flex-start queued resources. If no name, shows all."""
    cache = get_cache()
    flex_entries = {k: v for k, v in cache.items() if v.get("kind") == "flex-start"}

    if not flex_entries:
        print("No flex-start entries found in cache.")
        return

    if name is not None:
        if name not in flex_entries:
            print(f"❌ [bold blue]{name}[/bold blue] not found in cache or is not a flex-start entry.")
            return
        flex_entries = {name: flex_entries[name]}

    table = Table("Name", "Zone", "Type", "QR State", "VM State", "Queued Resource ID")
    has_suspended = False
    for node_id, instance in flex_entries.items():
        zone = instance["zone"]
        qr_id = instance["queued_resource_id"]
        try:
            info = describe_queued_resource(qr_id, zone)
            raw_state = info.get("state", {})
            state = raw_state.get("state", "UNKNOWN") if isinstance(raw_state, dict) else str(raw_state)
        except Exception:
            state = "ERROR"
        qr_color = _STATE_COLORS.get(state, "white")

        if state == "SUSPENDED":
            has_suspended = True

        if state == "ACTIVE":
            try:
                vm_state = get_state(node_id, zone)
            except Exception:
                vm_state = "UNKNOWN"
            vm_color = "bold green" if vm_state == "READY" else "yellow"
            vm_cell = f"[{vm_color}]{vm_state}[/{vm_color}]"
        else:
            vm_cell = "-"

        table.add_row(
            node_id, zone, instance["type"],
            f"[{qr_color}]{state}[/{qr_color}]",
            vm_cell,
            qr_id,
        )

    Console().print(table)

    if has_suspended:
        print("\nSuspended queued resources detected, running cleanup...")
        flex_cleanup()


@app.command()
def flex_cleanup():
    """Delete suspended queued resources from GCP and remove them from cache."""
    cache = get_cache()
    flex_entries = {k: v for k, v in cache.items() if v.get("kind") == "flex-start"}

    if not flex_entries:
        print("No flex-start entries found in cache.")
        return

    removed = []
    for node_id, instance in flex_entries.items():
        zone = instance["zone"]
        qr_id = instance["queued_resource_id"]
        try:
            info = describe_queued_resource(qr_id, zone)
            raw_state = info.get("state", {})
            state = (
                raw_state.get("state", "UNKNOWN")
                if isinstance(raw_state, dict)
                else str(raw_state)
            )
        except Exception:
            state = "ERROR"

        if state != "SUSPENDED":
            continue

        try:
            _run(
                f"gcloud alpha compute tpus queued-resources delete"
                f" {qr_id} --zone {zone} --quiet"
            )
        except subprocess.CalledProcessError:
            print(
                f"❌ Could not delete queued resource"
                f" [bold blue]{qr_id}[/bold blue]."
            )
            continue

        del cache[node_id]
        removed.append(node_id)
        print(
            f"✅ Removed [bold blue]{node_id}[/bold blue]"
            f" (queued resource: {qr_id})"
        )

    if removed:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    else:
        print("No suspended queued resources to clean up.")


@app.command()
def print_config():
    """Show current config and cache file paths."""
    print("[bold green]Printing configuration[bold green]")
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Config file not found at {CONFIG_FILE}, create it first.")
        return
    else:
        print(f"Config file found at {CONFIG_FILE}")
    config = get_config()
    print(f"TPU name prefix: {config.tpu_name_prefix}")
    print(f"Extra startup script: {config.extra_startup_script}")
    print(f"SSH identity file: {config.ssh_identity_file}")
    if os.path.exists(CACHE_FILE):
        print(f"Cache file found at {CACHE_FILE}")
    else:
        print(f"❌ Cache file not found at {CACHE_FILE}")
        return


@app.command()
def cleanup_ssh_hosts(name: str | None = None):
    """Remove stale known_hosts entries for a TPU. If no name, cleans all cached."""
    cache = get_cache()
    if name is not None:
        cleanup_known_hosts(name)
    else:
        for element in cache:
            cleanup_known_hosts(element)
    print("✅ Done! Known_hosts cleaned up")


if __name__ == "__main__":
    app()

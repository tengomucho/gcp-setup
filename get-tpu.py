import getpass
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta

import typer
from rich import print
from rich.console import Console
from rich.table import Table

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.expanduser("~/.get-tpu")
CACHE_FILE = os.path.join(CONFIG_DIR, "cache.json")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
ZONES_CACHE_FILE = os.path.join(CONFIG_DIR, "zones-cache.json")
VERBOSE = os.getenv("VERBOSE", "0") == "1"

DEFAULT_ACCELERATOR = "v6e-4"
DEFAULT_SOFTWARE_VERSION = "v2-alpha-tpuv6e"

# Timeouts for the ssh/scp steps of an install. The install script waits up to
# APT_LOCK_TIMEOUT (see setup.sh) for unattended-upgrades to release the dpkg
# lock, so the remote-run budget has to be comfortably larger than that.
SCP_TIMEOUT = 300
REMOTE_INSTALL_TIMEOUT = 2700

# How long to keep trying when a node stops answering on port 22 while staying
# READY. Outages observed on 2026-08-04 ran from 17 to ~60 minutes, so this will
# not ride most of them out — that is deliberate. Five minutes is enough to tell
# a slow start from a real problem, and failing with the step name beats sitting
# there for half an hour.
UNREACHABLE_BUDGET = 300

# Where the detached install writes its output on the TPU, and the markers used
# to tell that output apart from gcloud's own chatter on the same stream.
REMOTE_LOG = "tpu-setup.log"
PAYLOAD_TAR = "get-tpu-payload.tar.gz"
LOG_TAG = "__TPULOG__"
RC_TAG = "__TPURC__"
ROT_TAG = "__TPUROT__"
# 2s per tick, so a follower session lasts at most ~5 min before handing control
# back to the local loop to re-check the overall deadline.
FOLLOW_SESSION_TICKS = 150

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
    "southamerica-west1-a",
]

app = typer.Typer()
_gcloud_auth_checked = False


@dataclass
class Config:
    tpu_name_prefix: str = "tpu-vm-"
    extra_startup_script: str | None = None
    ssh_identity_file: str | None = None


class GcloudAuthError(Exception):
    """gcloud is missing or has no active account.

    Kept distinct from RuntimeError so the retry loops (_retry_transient,
    wait_for_ssh_auth) don't catch it: a missing login never fixes itself,
    so retrying it for the whole budget just delays the clear error.
    """


def ensure_gcloud_authenticated():
    """Raise a clear error unless gcloud has an active account configured."""
    global _gcloud_auth_checked
    if _gcloud_auth_checked:
        return

    try:
        result = subprocess.run(
            [
                "gcloud",
                "auth",
                "list",
                "--filter=status:ACTIVE",
                "--format=value(account)",
            ],
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise GcloudAuthError("❌ gcloud is not installed or is not on PATH.") from None

    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip()
        message = "❌ No active gcloud account. Run `gcloud auth login` and try again."
        if detail:
            message = f"{message}\n   gcloud reported: {detail}"
        raise GcloudAuthError(message)

    _gcloud_auth_checked = True


def _run(cmd: str, timeout: int | None = None):
    """Run a command, streaming its output live, and raise on failure or timeout.

    Output is deliberately not captured: a remote apt waiting on the dpkg lock,
    or a setup script mid-install, must be visible while it runs. Capturing it
    turned every stall into a silent, unbounded hang.
    """
    if VERBOSE:
        print(f"[bold blue]Running command:[/bold blue] {cmd}")
    split_cmd = shlex.split(cmd)
    if split_cmd and split_cmd[0] == "gcloud":
        ensure_gcloud_authenticated()
    try:
        result = subprocess.run(split_cmd, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"❌ Timed out after {timeout}s running: {cmd}") from None
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, split_cmd)


def _gcloud_output(cmd: str) -> str:
    """Run a gcloud command whose parsed output we consume, returning only stdout.

    std/stderr are kept separate because every caller feeds the result to
    json.loads or strips quotes from it; a gcloud warning on stderr used to get
    concatenated in and silently break that parse. The nonzero status still
    surfaces through CalledProcessError, and stderr is echoed for the VERBOSE
    case so a failure is still diagnosable.
    """
    ensure_gcloud_authenticated()
    result = subprocess.run(shlex.split(cmd), text=True, capture_output=True)
    if result.returncode != 0:
        if VERBOSE:
            print(f"[bold red]gcloud stderr:[/bold red] {result.stderr.strip()}")
        # Attach stderr so callers can tell a resource-not-found from a
        # transient API failure rather than re-running gcloud to find out.
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )
    return result.stdout


def get_cache():
    cache_path = CACHE_FILE
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def save_cache(cache: dict):
    """Persist the cache, creating ~/.get-tpu if it does not exist."""
    if not os.access(CONFIG_DIR, os.F_OK):
        os.makedirs(CONFIG_DIR)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _zone_sort_key(zone: str) -> tuple[int, str]:
    """Sort europe first, then us, then everything else — the same order LOCATIONS uses."""
    if zone.startswith("europe-"):
        return (0, zone)
    if zone.startswith("us-"):
        return (1, zone)
    return (2, zone)


def discover_zones(accelerator_type: str) -> list[str]:
    """Sweep every GCP zone and return the ones that offer accelerator_type.

    Runs `gcloud compute tpus locations list` once, then fans out 16 parallel
    `accelerator-types list` calls. The result is merged into zones-cache.json
    under the accelerator-type key so refreshing one type never wipes another.
    """
    out = _gcloud_output("gcloud compute tpus locations list --format=json")
    all_zones = sorted(
        (loc["locationId"] for loc in json.loads(out)), key=_zone_sort_key
    )

    def _offers(zone: str) -> str | None:
        try:
            out = _gcloud_output(
                f"gcloud compute tpus accelerator-types list"
                f" --zone={zone} --format=json"
            )
            types = {t["type"] for t in json.loads(out)}
            return zone if accelerator_type in types else None
        except Exception:
            return None

    found = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for result in pool.map(_offers, all_zones):
            if result is not None:
                found.append(result)

    zones_cache = {}
    if os.path.exists(ZONES_CACHE_FILE):
        with open(ZONES_CACHE_FILE, "r") as f:
            zones_cache = json.load(f)
    zones_cache[accelerator_type] = found
    if not os.access(CONFIG_DIR, os.F_OK):
        os.makedirs(CONFIG_DIR)
    with open(ZONES_CACHE_FILE, "w") as f:
        json.dump(zones_cache, f, indent=2)

    return found


def get_zones(accelerator_type: str, rediscover: bool = False) -> list[str]:
    """Return the cached zone list for accelerator_type, sweeping only when needed.

    A missing entry triggers discovery on its own; the flag is only needed to
    refresh a type that's already cached.
    """
    if os.path.exists(ZONES_CACHE_FILE) and not rediscover:
        with open(ZONES_CACHE_FILE, "r") as f:
            zones_cache = json.load(f)
        if accelerator_type in zones_cache:
            return zones_cache[accelerator_type]
    return discover_zones(accelerator_type)


@app.command("discover-zones")
def discover_zones_cmd(
    accelerator_type: str = DEFAULT_ACCELERATOR,
    force: bool = False,
):
    """Discover which GCP zones offer a given accelerator type, caching the result."""
    if force:
        print(f"🔄 Force-refreshing zone cache for [bold]{accelerator_type}[/bold]...")
        zones = discover_zones(accelerator_type)
        print(f"✅ Found {len(zones)} zones offering {accelerator_type}:")
        for z in zones:
            print(f"  {z}")
        return

    if os.path.exists(ZONES_CACHE_FILE):
        with open(ZONES_CACHE_FILE, "r") as f:
            zones_cache = json.load(f)
        if accelerator_type in zones_cache:
            zones = zones_cache[accelerator_type]
            print(f"📋 {accelerator_type} zones (from cache):")
            for z in zones:
                print(f"  {z}")
            print(f"\nRun with --force to refresh.")
            return

    print(f"🔍 Discovering zones offering [bold]{accelerator_type}[/bold]...")
    zones = discover_zones(accelerator_type)
    print(f"✅ Found {len(zones)} zones offering {accelerator_type}:")
    for z in zones:
        print(f"  {z}")


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
        "\nYou can define a path to a script that stages extra files into the "
        "install payload (called with a staging directory as its only arg). "
        "Press return to leave empty",
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
    value = _gcloud_output("gcloud config get-value project --format=json")
    return value.replace('"', "").strip()


def list_tpus(zone: str):
    desc = _gcloud_output(
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


def wait_for_ssh(
    name: str, zone: str, timeout: int = UNREACHABLE_BUDGET, interval: int = 5
):
    """Poll the TPU until SSH accepts connections on its external IP.

    The node may not have an external IP yet (or may not even appear in
    `tpu-vm list`): a flex-start winner is only materialised once its queued
    resource turns PROVISIONING, and the guest agent can lag behind ACTIVE.
    Fetching the IP inside the loop instead of assuming it exists lets this one
    routine cover both the normal create path and a freshly-won flex race, and
    sidesteps gcloud's internal 10x5s retry loop with no connect timeout, which
    otherwise reads as a multi-minute hang.

    A node that has been up for hours can also stop answering here. We wait out
    the short version and give up on the rest (see _retry_transient).
    """
    deadline = time.time() + timeout
    next_notice = time.time() + 60
    ext_ip = None
    while time.time() < deadline:
        if ext_ip is None:
            try:
                ext_ip = get_ext_ip(name, zone)
                print(f"⏳ Waiting for SSH to become reachable on {ext_ip}:22...")
            except Exception:
                # Not materialised / no external IP assigned yet — keep waiting.
                ext_ip = None
        if ext_ip is not None:
            try:
                with socket.create_connection((ext_ip, 22), timeout=5):
                    print("✅ SSH port is open.")
                    return
            except OSError:
                pass
        if time.time() >= next_notice:
            target = f"{ext_ip}:22" if ext_ip else f"{name} (no external IP yet)"
            print(
                f"   still no answer on {target},"
                f" {int(deadline - time.time())}s of patience left..."
            )
            next_notice = time.time() + 60
        time.sleep(interval)
    if ext_ip is None:
        raise RuntimeError(
            f"❌ {name} got no external IP within {timeout}s."
            f" Check the node's state before retrying."
        )
    raise RuntimeError(
        f"❌ No answer on {ext_ip}:22 after {timeout}s. The node reports ready but is"
        f" not accepting SSH — check its state before retrying."
    )


def wait_for_ssh_auth(
    name: str,
    zone: str,
    project: str,
    budget: int = UNREACHABLE_BUDGET,
    interval: int = 10,
):
    """Confirm a real SSH session can be opened, not just that port 22 answers.

    sshd can be listening before the guest agent has propagated the pushed key.
    Failing here explicitly beats handing the problem to gcloud, which retries
    10x5s internally and reports nothing useful. Bounded by the same budget as
    every other reachability wait, so the whole phase cannot outlive it.
    """
    deadline = time.time() + budget
    attempt = 0
    while True:
        attempt += 1
        remaining = int(deadline - time.time())
        try:
            _run(
                _ssh_command(name, zone, project, "true"),
                timeout=max(1, min(60, remaining)),
            )
            print("✅ SSH authentication works.")
            return
        except (subprocess.CalledProcessError, RuntimeError):
            if time.time() + interval >= deadline:
                raise RuntimeError(
                    f"❌ Gave up on: opening an SSH session to {name}.\n"
                    f"   Failed {attempt}x over {budget}s. Either the pushed key is"
                    f" not propagated or the node is not reachable from here."
                ) from None
            print(
                f"⏳ SSH not usable yet (attempt {attempt}), retrying in {interval}s"
                f" ({remaining}s left before giving up)..."
            )
            time.sleep(interval)


def _retry_transient(label: str, budget: int, fn):
    """Retry fn until it succeeds or budget runs out, then fail naming the step.

    A TPU's external IP can stop answering on port 22 while the node stays READY,
    sshd keeps listening and ICMP keeps working. Brief hiccups are worth retrying;
    anything still failing after the budget is a real problem, and stopping with
    the step name is more useful than waiting it out.

    fn is called with the seconds left in the budget and must bound itself by that,
    so no single attempt can overrun the deadline.
    """
    deadline = time.time() + budget
    attempt = 0
    while True:
        attempt += 1
        remaining = deadline - time.time()
        try:
            return fn(max(1, int(remaining)))
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            backoff = 10
            if time.time() + backoff >= deadline:
                raise RuntimeError(
                    f"❌ Gave up on: {label}.\n"
                    f"   Failed {attempt}x over {budget}s, so this is not a slow"
                    f" start. Last error: {exc}"
                ) from None
            print(
                f"⚠️  {label} failed (attempt {attempt}), retrying in {backoff}s"
                f" ({int(deadline - time.time())}s left before giving up)..."
            )
            time.sleep(backoff)


def _ssh_command(name: str, zone: str, project: str, remote_cmd: str) -> str:
    """Build a gcloud tpu-vm ssh invocation running remote_cmd."""
    return (
        f"gcloud compute tpus tpu-vm ssh --zone {zone} {name} --project {project}"
        f" --ssh-flag=-o --ssh-flag=ConnectTimeout=10"
        f" --ssh-flag=-o --ssh-flag=ServerAliveInterval=15"
        f" --ssh-flag=-o --ssh-flag=ServerAliveCountMax=4"
        f" --command={shlex.quote(remote_cmd)}"
    )


def remote_run_logged(
    name: str,
    zone: str,
    project: str,
    script: str,
    log: str = REMOTE_LOG,
    timeout: int = REMOTE_INSTALL_TIMEOUT,
    prepare: str = "",
):
    """Run a script on the TPU detached from the SSH channel, following its log.

    The install takes 10-25 minutes and used to run in the foreground of a single
    ssh channel, so anything that dropped that channel — an sshd restart from
    unattended-upgrades, a laptop sleep, a flaky link — killed the install with
    no log left behind. Here the script is launched under setsid/nohup writing to
    ~/{log}, and a second session merely tails it. Losing the tail costs nothing:
    we reattach at the line we got to, and the install keeps going regardless.
    """
    rc_file = f"{log}.rc"
    pid_file = f"{log}.pid"
    # Re-attach instead of restarting if a previous invocation left one running.
    # Liveness comes from a pid file rather than pgrep: the launch snippet has
    # "bash {script}" in its own argv, so pgrep -f would always match itself.
    # `prepare` only runs when we actually launch, never when re-attaching: it
    # would otherwise overwrite the files out from under a running install.
    launch = (
        f"cd ~;"
        f" if [ -f {pid_file} ] && kill -0 \"$(cat {pid_file})\" 2>/dev/null; then"
        f" echo 'install already running, attaching to its log';"
        f" else {prepare} rm -f {rc_file}; : > {log};"
        f" setsid nohup sh -c 'echo $$ >{pid_file};"
        f" bash {script} >{log} 2>&1; echo $? >{rc_file}'"
        f" </dev/null >/dev/null 2>&1 & fi"
    )
    print(f"🏃 Running {script} on the TPU (detached, log: ~/{log})")
    _retry_transient(
        f"launching {script} on {name}",
        UNREACHABLE_BUDGET,
        lambda left: _run(_ssh_command(name, zone, project, launch), timeout=left),
    )

    # Log lines are tagged remotely so gcloud's own chatter on stderr cannot be
    # mistaken for install output and throw off the resume offset.
    offset = 1
    attempt = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Each session is capped so control always comes back here to re-check the
        # deadline; an uncapped follower would be its own silent hang. It signals
        # why it ended: ROT_TAG for hitting the cap (reattach quietly), RC_TAG for
        # the script finishing, neither for a dropped channel.
        #
        # A polling tailer rather than `tail -F`: every command runs in the
        # foreground and the loop is bounded, so the session always reaches EOF on
        # its own. A backgrounded `tail -F` kept the pipe open after its watcher
        # was killed and blocked the local read indefinitely.
        follow = (
            f"cd ~; n={offset}; i=0;"
            f" while :; do"
            f" t=$(wc -l <{log} 2>/dev/null | tr -d ' ' || echo 0);"
            f' if [ "$t" -ge "$n" ]; then'
            f" tail -n +$n {log} | sed 's/^/{LOG_TAG}/'; n=$((t+1)); fi;"
            f" if [ -f {rc_file} ]; then break; fi;"
            f" i=$((i+1));"
            f" if [ $i -ge {FOLLOW_SESSION_TICKS} ]; then echo {ROT_TAG}; break; fi;"
            f" sleep 2; done;"
            f" [ -f {rc_file} ] && printf '{RC_TAG}%s\\n' \"$(cat {rc_file})\"; true"
        )
        cmd = _ssh_command(name, zone, project, follow)
        ensure_gcloud_authenticated()
        if VERBOSE:
            print(f"[bold blue]Running command:[/bold blue] {cmd}")
        proc = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None  # stdout=PIPE always gives us one
        rc = None
        rotated = False
        for line in proc.stdout:
            if line.startswith(RC_TAG):
                rc = int(line[len(RC_TAG) :].strip() or 1)
                break
            if line.startswith(ROT_TAG):
                rotated = True
                continue
            if line.startswith(LOG_TAG):
                offset += 1
                line = line[len(LOG_TAG) :]
            print(line, end="")
        proc.stdout.close()
        proc.terminate()
        proc.wait()

        if rc == 0:
            return
        if rc is not None:
            raise RuntimeError(
                f"❌ {script} exited {rc} on {name}."
                f" Full log: ssh {name} 'cat ~/{log}'"
            )
        if rotated:
            # Ran its full course: a normal session rotation, not a failure.
            attempt = 0
            continue
        attempt += 1
        backoff = min(60, 5 * 2 ** (attempt - 1))
        print(
            f"⚠️  Lost the log stream (attempt {attempt}); the install is still"
            f" running on the TPU. Reattaching at line {offset} in {backoff}s..."
        )
        time.sleep(backoff)

    raise RuntimeError(
        f"❌ {script} did not finish within {timeout}s."
        f" It may still be running: ssh {name} 'tail -f ~/{log}'"
    )


def describe_queued_resource(queued_resource_id: str, zone: str) -> dict:
    out = _gcloud_output(
        f"gcloud alpha compute tpus queued-resources describe"
        f" {queued_resource_id} --zone {zone} --format json"
    )
    return json.loads(out)


def queued_resource_state(queued_resource_id: str, zone: str) -> str:
    """Return the queued resource's state, or GONE if GCP no longer has it.

    GONE and ERROR are deliberately distinct: a cache entry may only be dropped
    when GCP actually reports the resource missing, never on a transient API
    failure that would otherwise look identical.
    """
    try:
        out = _gcloud_output(
            f"gcloud alpha compute tpus queued-resources describe"
            f" {queued_resource_id} --zone {zone} --format json"
        )
        info = json.loads(out)
    except subprocess.CalledProcessError as exc:
        # A describe of a deleted resource exits non-zero with NOT_FOUND on
        # stderr. That is how a cancelled/expired request reaches the caller;
        # a genuine API failure must stay distinct from "gone".
        stderr = (exc.stderr or "").lower()
        if "not_found" in stderr or "not found" in stderr:
            return "GONE"
        return "ERROR"
    raw_state = info.get("state", {})
    if isinstance(raw_state, dict):
        return raw_state.get("state", "UNKNOWN")
    return str(raw_state)


def qr_state(info: dict) -> str:
    """Pull the queued resource state string out of a `describe` result.

    GCP returns `state` as either a dict with a `state` key (annotated form)
    or a bare string, depending on which endpoint/format produced it.
    """
    raw_state = info.get("state", {})
    if isinstance(raw_state, dict):
        return raw_state.get("state", "UNKNOWN")
    return str(raw_state)


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
    # And point the tpu-hermes profile at the fresh alias, if it's set up.
    _write_hermes_env(name)


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


def build_payload(tmpdir: str, config: Config) -> str:
    """Collect everything the install needs into a single tarball.

    Shipping one archive replaces the nine separate scp/ssh invocations this used
    to take, and lets the whole install run as one remote process.
    """
    stage = os.path.join(tmpdir, "payload")
    os.makedirs(stage)
    for script in ("setup.sh", "run-all.sh"):
        shutil.copy(os.path.join(CUR_DIR, script), stage)

    if config.extra_startup_script:
        extra = os.path.join(stage, "extra")
        os.makedirs(extra)
        print(f"🔧 Staging extra files with {config.extra_startup_script}")
        subprocess.check_call(
            f"{config.extra_startup_script} {shlex.quote(extra)}", shell=True
        )

    tar_path = os.path.join(tmpdir, PAYLOAD_TAR)
    # COPYFILE_DISABLE keeps bsdtar on macOS from adding ._* AppleDouble entries.
    _run(f"env COPYFILE_DISABLE=1 tar czf {tar_path} -C {stage} .", timeout=SCP_TIMEOUT)
    return tar_path


def install_tpu_script(name: str, location: str, project: str, config: Config):
    wait_for_ssh(name, location)
    wait_for_ssh_auth(name, location, project)
    print("🤖 Retrieving IP and updating local ssh settings")
    update_ssh_config(name, location)

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = build_payload(tmpdir, config)
        print("🧾 Copying the install payload")
        _retry_transient(
            f"copying the install payload to {name}",
            UNREACHABLE_BUDGET,
            lambda left: _run(
                f"gcloud compute tpus tpu-vm scp --zone {location}"
                f" --scp-flag=-o --scp-flag=ConnectTimeout=10"
                f" {tar_path} {name}:{PAYLOAD_TAR} --project {project}",
                timeout=left,
            ),
        )

    remote_run_logged(
        name,
        location,
        project,
        "run-all.sh",
        prepare=f"tar xzf {PAYLOAD_TAR} || exit 1;",
    )
    print(f"✅ Done! You can now use [bold green]{name}[/bold green]")


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
    accelerator_type: str = DEFAULT_ACCELERATOR,
    software_version: str = DEFAULT_SOFTWARE_VERSION,
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
            save_cache(cache)
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

    for tpu_name in cache:
        instance = cache[tpu_name]
        zone = instance["zone"]
        print(f"\nChecking [bold blue]{tpu_name}[/bold blue] in [bold]{zone}[/bold]...")
        try:
            restart_tpu(tpu_name, zone)
            return
        except subprocess.CalledProcessError:
            print(f"❌ TPU [bold blue]{tpu_name}[/bold blue] is not available")
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
    for tpu_name in cache:
        instance = cache[tpu_name]
        zone = instance["zone"]
        print(f"\nChecking [bold blue]{tpu_name}[/bold blue] in [bold]{zone}[/bold]...")
        state = get_state(tpu_name, zone)
        if state == "READY":
            print(
                f"Stopping TPU [bold blue]{tpu_name}[/bold blue] in [bold]{zone}[/bold]..."
            )
            _run(f"gcloud compute tpus tpu-vm stop {tpu_name} --zone {zone}")
            print(f"🧘 TPU [bold blue]{tpu_name}[/bold blue] stopped")
            return
        else:
            print(
                f"TPU {tpu_name} is not running, (state: [cyan]{state}[/cyan]) skipping.."
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
    save_cache(cache)
    print(f"✅ TPU [bold blue]{name}[/bold blue] deleted")
    print("[bold orange]Note:[/bold orange] check if disks need to be deleted too.")


@app.command()
def flex_start(
    zone: str,
    accelerator_type: str = DEFAULT_ACCELERATOR,
    software_version: str = DEFAULT_SOFTWARE_VERSION,
    max_run_duration: str = "9h",
    auto_reinstall: bool = typer.Option(False, "--reinstall", "-r", help="Poll every 5 s and run reinstall when ACTIVE"),
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
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        if "already exists" in stderr.lower():
            print(
                f"\n⚠️  A queued resource named [bold blue]{queued_resource_id}[/bold blue] already exists"
                f" in [bold]{zone}[/bold] on GCP, but it wasn't in the local cache"
                f" (someone likely created it outside this tool, or the cache was reset)."
            )
            print(
                f"   Re-adding it to the local cache. Run [bold]flex-status[/bold] to check its state and\n"
                f"   delete it properly."
            )
            cache[node_id] = {
                "type": accelerator_type,
                "zone": zone,
                "queued_resource_id": queued_resource_id,
                "kind": "flex-start",
            }
            save_cache(cache)
        else:
            print(f"❌ Failed to submit flex-start request for [bold]{zone}[/bold]")
        return

    cache[node_id] = {
        "type": accelerator_type,
        "zone": zone,
        "queued_resource_id": queued_resource_id,
        "kind": "flex-start",
    }
    save_cache(cache)

    print(
        f"\n✅ Queued resource [bold blue]{queued_resource_id}[/bold blue] submitted."
        f" Use [bold]flex-status[/bold] to monitor its state."
    )

    if auto_reinstall:
        print(f"\n[bold]Polling every 5 s for [bold blue]{node_id}[/bold blue] to become ACTIVE...[/bold]")
        start_time = time.time()
        while True:
            time.sleep(5)
            try:
                info = describe_queued_resource(queued_resource_id, zone)
                state = qr_state(info)
            except Exception:
                state = "ERROR"

            color = _STATE_COLORS.get(state, "white")
            waited = timedelta(seconds=int(time.time() - start_time))
            print(
                f"  [{color}]{state}[/{color}] ({node_id}) - started {waited} ago"
            )

            if state == "ACTIVE":
                elapsed = time.time() - start_time
                print(f"\n✅ Resource is ACTIVE after {elapsed:.1f} secs. Starting reinstall...")
                reinstall(node_id)
                break
            elif state in ("SUSPENDED", "FAILED", "ERROR"):
                print(f"\n❌ Resource entered terminal state [{color}]{state}[/{color}], aborting auto-reinstall.")
                break


_STATE_COLORS = {
    "ACTIVE": "bold green",
    "WAITING_FOR_RESOURCES": "yellow",
    "PROVISIONING": "yellow",
    "FAILED": "bold red",
    "SUSPENDING": "red",
    "SUSPENDED": "red",
}

# ---------------------------------------------------------------------------
# flex-race
# ---------------------------------------------------------------------------

RACE_POLL_INTERVAL = 10
# After the losers are cancelled the winner still has to spin up: PROVISIONING
# means flex capacity was found, but the node may not have an external IP (or
# even appear in `tpu-vm list`) yet. Reinstall wants a real VM, so we keep
# watching until it turns ACTIVE. This bounds that wait.
RACE_ACTIVE_WAIT = 900


def _race_verdict(states: dict[str, str]) -> tuple[str | None, bool]:
    """Return (winner, all_dead) from a {node_id: state} mapping.

    Winner: the first node in PROVISIONING or ACTIVE. all_dead: every
    participant is in a terminal state (FAILED / SUSPENDED / GONE / ERROR).
    """
    winner = None
    for node_id, state in states.items():
        if state in ("PROVISIONING", "ACTIVE") and winner is None:
            winner = node_id
    all_dead = all(
        s in ("FAILED", "SUSPENDED", "GONE", "ERROR") for s in states.values()
    )
    return winner, all_dead


def _wait_for_winner_active(
    winner: str, cache: dict, timeout: int = RACE_ACTIVE_WAIT
) -> bool:
    """Block until the winner's node turns ACTIVE; return False on failure.

    Cancelling the losers the moment the winner turns PROVISIONING is what
    saves the second bill, but PROVISIONING is not 'installable': the node may
    not have been handed out an external IP yet, and `tpu-vm list` may not even
    show it. Reinstall wants a real, reachable VM, so after the losers are
    cancelled we keep polling just the winner until GCP materialises it.
    FAILED / SUSPENDED / GONE abort early; a transient read ERROR is treated
    as 'keep waiting' rather than losing track of a live node.
    """
    zone = cache[winner]["zone"]
    started = time.time()
    print(f"\n⏳ Waiting for [bold blue]{winner}[/bold blue] to become ACTIVE...")
    while time.time() - started < timeout:
        time.sleep(RACE_POLL_INTERVAL)
        state = queued_resource_state(winner, zone)
        elapsed = int(time.time() - started)
        if state == "ACTIVE":
            print(f"✅ [bold green]{winner}[/bold green] is ACTIVE after {elapsed}s.")
            return True
        if state in ("FAILED", "SUSPENDED", "GONE"):
            print(
                f"❌ [bold red]{winner}[/bold red] entered terminal state"
                f" [{state}], aborting."
            )
            return False
        print(f"   {winner} is {state}, waiting for ACTIVE ({elapsed}s)...")
    print(
        f"❌ [bold red]{winner}[/bold red] did not reach ACTIVE within"
        f" {timeout}s."
    )
    return False


def _submit_flex(
    node_id: str, zone: str, accel: str, version: str, duration: str
) -> str | None:
    """Submit a flex-start request, returning None on success or a short reason."""
    cmd = (
        f"gcloud alpha compute tpus queued-resources create {node_id}"
        f" --zone={zone}"
        f" --accelerator-type={accel}"
        f" --runtime-version={version}"
        f" --node-id={node_id}"
        f" --provisioning-model=flex-start"
        f" --max-run-duration={duration}"
    )
    try:
        _gcloud_output(cmd)
        return None
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        # Take the last non-empty line as the reason — it's usually the one
        # that carries the actual error message.
        for line in reversed(stderr.splitlines()):
            if line.strip():
                return line.strip()
        return f"exit code {exc.returncode}"


def _race_table(
    states: dict[str, str], failures: dict[str, str], started_at: float
) -> Table:
    """Build the rich renderable for the live race view."""
    waiting = sum(
        1 for s in states.values() if s not in ("FAILED", "SUSPENDED", "GONE", "ERROR")
    )
    elapsed = timedelta(seconds=int(time.time() - started_at))
    table = Table(
        title=f"requested {len(states)} | waiting {waiting} | elapsed {elapsed}",
        title_justify="left",
    )
    table.add_column("Zone")
    table.add_column("State")
    for node_id, state in states.items():
        zone = node_id.split("flex-")[-1]
        color = _STATE_COLORS.get(state, "white")
        table.add_row(zone, f"[{color}]{state}[/{color}]")
    if failures:
        table.add_row(
            f"[dim]+{len(failures)} refused[/dim]",
            "[dim]see below[/dim]",
        )
    return table


@app.command("flex-race")
def flex_race(
    max_run_duration: str = typer.Argument("8h"),
    accelerator_type: str = DEFAULT_ACCELERATOR,
    software_version: str = DEFAULT_SOFTWARE_VERSION,
    zone_discovery: bool = False,
):
    """Fan out flex-start requests to every zone offering the accelerator type.

    The first zone to reach PROVISIONING wins; the rest are cancelled and, once
    the winner turns ACTIVE, the normal install runs on it. Queued requests are
    not billed while waiting, so fanning out costs nothing but the
    cancellation bookkeeping.
    """
    from rich.live import Live

    ensure_gcloud_authenticated()

    zones = get_zones(accelerator_type, rediscover=zone_discovery)
    if not zones:
        print(f"❌ No zones found offering {accelerator_type}.")
        return
    if zone_discovery:
        print(f"🔄 Zone list refreshed: {len(zones)} zones offering {accelerator_type}.")
    else:
        print(f"📋 Using cached zone list: {len(zones)} zones offering {accelerator_type}.")

    config = get_config()
    cache = get_cache()
    project = get_project()

    # -- submit ---------------------------------------------------------------
    print(f"\n[bold green]Submitting flex-start requests to {len(zones)} zones...[/bold green]")
    states: dict[str, str] = {}
    failures: dict[str, str] = {}

    def _submit_one(zone: str) -> tuple[str, str, str | None]:
        node_id = f"{config.tpu_name_prefix}flex-{zone}"
        reason = _submit_flex(
            node_id, zone, accelerator_type, software_version, max_run_duration
        )
        return node_id, zone, reason

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_submit_one, zones))

    accepted = []
    for node_id, zone, reason in results:
        if reason is None:
            states[node_id] = "SUBMITTED"
            accepted.append((node_id, zone))
        else:
            failures[zone] = reason

    # One cache write for all accepted entries so flex-status / flex-cancel /
    # flex-cleanup all keep working on them.
    for node_id, zone in accepted:
        cache[node_id] = {
            "type": accelerator_type,
            "zone": zone,
            "queued_resource_id": node_id,
            "kind": "flex-start",
        }
    save_cache(cache)

    print(f"✅ Submitted to {len(accepted)} zones; {len(failures)} refused.")
    if failures:
        for zone, reason in sorted(failures.items()):
            print(f"   {zone}: {reason}")

    if not states:
        print("❌ No zone accepted the request. Nothing to race.")
        return

    # -- poll loop ------------------------------------------------------------
    started_at = time.time()
    winner: str | None = None

    def _poll_all():
        """Fetch the state of every live participant in parallel."""
        def _one(node_id: str) -> tuple[str, str]:
            try:
                info = describe_queued_resource(node_id, cache[node_id]["zone"])
                return node_id, qr_state(info)
            except Exception:
                return node_id, "ERROR"

        with ThreadPoolExecutor(max_workers=8) as pool:
            for node_id, state in pool.map(_one, list(states)):
                states[node_id] = state

    try:
        with Live(_race_table(states, failures, started_at), refresh_per_second=1) as live:
            while True:
                time.sleep(RACE_POLL_INTERVAL)
                _poll_all()
                winner, all_dead = _race_verdict(states)
                live.update(_race_table(states, failures, started_at))
                if winner or all_dead:
                    break
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — cancelling all submitted requests...")
        _cancel_all(states, cache)
        flex_cleanup()
        print("Done.")
        return

    # -- winner or all dead ---------------------------------------------------
    if all_dead and not winner:
        print("\n❌ All requests ended in a terminal state. No winner.")
        flex_cleanup()
        return

    assert winner is not None  # _race_verdict guarantees this when not all_dead
    print(f"\n🏆 [bold green]{winner}[/bold green] is {states[winner]}! Cancelling the rest...")
    losers = [n for n in states if n != winner]
    with ThreadPoolExecutor(max_workers=8) as pool:
        pool.map(lambda n: _delete_queued_resource(n, cache[n]["zone"], force=True), losers)
    flex_cleanup()

    # -- wait for the winner to be installable --------------------------------
    # PROVISIONING beats the losers to the cancel, but the node may not be
    # reachable (or even listed) until it turns ACTIVE. Don't hand a phantom to
    # the installer.
    if not _wait_for_winner_active(winner, cache):
        print("No install to run — leaving the winner node as-is.")
        return

    # -- install --------------------------------------------------------------
    print(f"\n🚀 Running install on [bold blue]{winner}[/bold blue]...")
    reinstall(winner)


def _cancel_all(states: dict[str, str], cache: dict):
    """Cancel every submitted request in parallel (best-effort)."""
    def _one(node_id: str):
        try:
            _delete_queued_resource(node_id, cache[node_id]["zone"], force=True)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=8) as pool:
        pool.map(_one, list(states))


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

    table = Table("Name", "Zone", "Type", "QR State", "VM State", "Requested")
    has_suspended = False
    for node_id, instance in flex_entries.items():
        zone = instance["zone"]
        qr_id = instance["queued_resource_id"]
        create_time = None
        try:
            info = describe_queued_resource(qr_id, zone)
            state = qr_state(info)
            create_time = info.get("createTime")
        except Exception:
            state = "ERROR"
        qr_color = _STATE_COLORS.get(state, "white")

        if create_time:
            # GCP returns nanosecond precision, which fromisoformat rejects.
            ts = re.sub(r"(\.\d{6})\d+", r"\1", create_time.replace("Z", "+00:00"))
            created = datetime.fromisoformat(ts)
            age = timedelta(seconds=int(time.time() - created.timestamp()))
            requested_cell = f"{created.astimezone().strftime('%Y-%m-%d %H:%M:%S')} ({age} ago)"
        else:
            requested_cell = "-"

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
            requested_cell,
        )

    Console().print(table)

    if has_suspended:
        print("\nSuspended queued resources detected, running cleanup...")
        flex_cleanup()


def _delete_queued_resource(qr_id: str, zone: str, force: bool = False):
    """Delete a queued resource, optionally with --force."""
    force_flag = " --force" if force else ""
    _run(
        f"gcloud alpha compute tpus queued-resources delete"
        f" {qr_id} --zone {zone}{force_flag} --quiet"
    )


@app.command()
def flex_cancel(
    name: str | None = typer.Argument(None, help="TPU name to cancel; defaults to all cached ones"),
):
    """Cancel a pending flex-start request. If no name, cancels all cached ones.

    There is no cancel verb for queued resources: deleting the request is how you
    withdraw it while it is still WAITING_FOR_RESOURCES, and it also tears down
    the node once one has been handed out. flex-cleanup runs afterwards to drop
    the cancelled entries from the cache, since a cancel that left them behind
    would keep the name blocked for the next flex-start.
    """
    cache = get_cache()
    flex_entries = {k: v for k, v in cache.items() if v.get("kind") == "flex-start"}

    if not flex_entries:
        print("No flex-start entries found in cache.")
        return

    if name is not None:
        if name not in flex_entries:
            print(
                f"❌ [bold blue]{name}[/bold blue] not found in cache or is not a"
                f" flex-start entry."
            )
            return
        flex_entries = {name: flex_entries[name]}

    def _cancel_one(instance: dict) -> tuple[str, str, str | None, bool]:
        """Cancel one flex-start entry. Returns (node_id, zone, error_reason, was_gone)."""
        node_id = instance["queued_resource_id"]
        zone = instance["zone"]
        state = queued_resource_state(node_id, zone)

        if state == "GONE":
            return node_id, zone, None, True

        try:
            _delete_queued_resource(node_id, zone, force=True)
            return node_id, zone, None, False
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            reason = stderr.splitlines()[-1] if stderr else f"exit code {exc.returncode}"
            return node_id, zone, reason, False

    entries = list(flex_entries.values())
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_cancel_one, entries))

    cancelled = []
    already_gone = []
    failed = []
    for node_id, zone, reason, was_gone in results:
        if reason is not None:
            failed.append((node_id, zone, reason))
        elif was_gone:
            already_gone.append(node_id)
        else:
            cancelled.append(node_id)

    for node_id in already_gone:
        print(f"[bold blue]{node_id}[/bold blue] is already gone from GCP, nothing to cancel.")
    for node_id in cancelled:
        print(f"✅ Cancelled [bold blue]{node_id}[/bold blue]")
    for node_id, zone, reason in failed:
        print(f"❌ Could not cancel [bold blue]{node_id}[/bold blue] in [bold]{zone}[/bold]: {reason}")

    print("\nReconciling the cache...")
    flex_cleanup()


@app.command()
def flex_cleanup():
    """Reconcile the cache with GCP, dropping suspended and vanished entries.

    Two cases need clearing and only one used to be handled. A SUSPENDED request
    still exists on GCP and has to be deleted there first. A cancelled or expired
    one is already gone, leaving just the cache entry — that fell through the old
    `state != "SUSPENDED": continue`, so flex-status kept reporting resources that
    no longer existed and the name stayed blocked for the next flex-start.
    """
    cache = get_cache()
    flex_entries = {k: v for k, v in cache.items() if v.get("kind") == "flex-start"}

    if not flex_entries:
        print("No flex-start entries found in cache.")
        return

    removed = []
    for node_id, instance in flex_entries.items():
        zone = instance["zone"]
        qr_id = instance["queued_resource_id"]
        state = queued_resource_state(qr_id, zone)

        if state == "ERROR":
            # Could be a transient API failure, so keep the entry rather than
            # lose track of a resource that may well still be running.
            print(
                f"⚠️  Could not read the state of [bold blue]{qr_id}[/bold blue],"
                f" leaving it in the cache."
            )
            continue

        if state == "SUSPENDED":
            try:
                _delete_queued_resource(qr_id, zone)
            except subprocess.CalledProcessError:
                print(
                    f"❌ Could not delete queued resource"
                    f" [bold blue]{qr_id}[/bold blue]."
                )
                continue
        elif state != "GONE":
            continue

        del cache[node_id]
        removed.append(node_id)
        print(
            f"✅ Removed [bold blue]{node_id}[/bold blue]"
            f" ({state.lower()}, queued resource: {qr_id})"
        )

    if removed:
        save_cache(cache)
    else:
        print("Nothing to clean up.")


# ---------------------------------------------------------------------------
# hermes-setup / hermes-remove
# ---------------------------------------------------------------------------

# Name of the dedicated Hermes profile used to drive a TPU over SSH. Not
# "tpu": `hermes profile create` drops a wrapper script in ~/.local/bin named
# after the profile, and a bare `tpu` command would shadow nothing but read
# like a TPU manager, which this is not.
HERMES_PROFILE = "tpu-hermes"


def _hermes_profile_exists() -> bool:
    """True if the tpu-hermes profile directory exists on disk."""
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return os.path.isdir(os.path.join(hermes_home, "profiles", HERMES_PROFILE))


def _hermes_profile_env_path() -> str:
    """Absolute path to the tpu-hermes profile's .env file."""
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return os.path.join(hermes_home, "profiles", HERMES_PROFILE, ".env")


def _write_hermes_env(host_alias: str):
    """Point the tpu-hermes profile at the given TPU, if the profile exists.

    Hermes's SSH backend reads TERMINAL_SSH_HOST/USER from the profile's .env
    when a session starts, so writing the file here is all a fresh
    `tpu-hermes` session needs to land on the new VM. We write the SSH *alias*
    (the Host entry update_ssh_config just refreshed) rather than the raw IP:
    the alias always resolves to the current address via ~/.ssh/config, so the
    .env survives the IP churn every stop/start brings. USER is the local
    account name, matching what update_ssh_config puts in the Host entry.

    A no-op when the profile is absent — hermes-setup is opt-in.
    """
    if not _hermes_profile_exists():
        return
    env_path = _hermes_profile_env_path()
    user = getpass.getuser()
    # Merge, don't clobber: the profile's .env may carry other keys (API
    # tokens the user added), so rewrite only the two lines we own.
    managed = {"TERMINAL_SSH_HOST": host_alias, "TERMINAL_SSH_USER": user}
    lines = []
    seen = set()
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                key = line.split("=", 1)[0].strip()
                if key in managed:
                    lines.append(f"{key}={managed[key]}\n")
                    seen.add(key)
                else:
                    lines.append(line)
    for key, value in managed.items():
        if key not in seen:
            lines.append(f"{key}={value}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)
    print(f"🔀 Pointed Hermes profile [bold blue]{HERMES_PROFILE}[/bold blue] at {host_alias}")


@app.command("hermes-setup")
def hermes_setup():
    """Create the tpu-hermes Hermes profile (SSH backend, persistent shell).

    The profile makes Hermes run its terminal and file tools on the TPU over
    SSH while the agent itself stays on this machine. The TPU address is NOT
    set here — it is written to the profile's .env (TERMINAL_SSH_HOST/USER) by
    create/restart once a VM exists.
    """
    if _hermes_profile_exists():
        print(f"✅ Hermes profile [bold blue]{HERMES_PROFILE}[/bold blue] already exists, nothing to do.")
        return

    if shutil.which("hermes") is None:
        print("❌ hermes is not installed or not on PATH — cannot create the profile.")
        raise typer.Exit(1)

    print(f"[bold green]Creating Hermes profile {HERMES_PROFILE}[bold green]")
    # --no-skills: the profile only needs to run commands on the TPU; skipping
    # the bundled skill set keeps it small and out of `hermes update`'s way.
    _run(f"hermes profile create {HERMES_PROFILE} --no-skills")
    # `hermes config set` writes the *active* profile, and HERMES_PROFILE is not
    # honored by it — the global `-p` flag is the only way to target another
    # profile without switching to it.
    for key, value in (
        ("terminal.backend", "ssh"),
        ("terminal.persistent_shell", "true"),
    ):
        subprocess.run(
            ["hermes", "-p", HERMES_PROFILE, "config", "set", key, value],
            check=True,
            capture_output=True,
            text=True,
        )
    print(f"✅ Profile [bold blue]{HERMES_PROFILE}[/bold blue] created (backend: ssh, persistent shell).")
    print("   Launch it with: tpu-hermes")


@app.command("hermes-remove")
def hermes_remove():
    """Delete the tpu-hermes Hermes profile and its ~/.local/bin wrapper."""
    if not _hermes_profile_exists():
        print(f"❌ Hermes profile [bold blue]{HERMES_PROFILE}[/bold blue] does not exist, nothing to remove.")
        return

    if shutil.which("hermes") is None:
        print("❌ hermes is not installed or not on PATH — cannot remove the profile.")
        raise typer.Exit(1)

    print(f"[bold green]Deleting Hermes profile {HERMES_PROFILE}[bold green]")
    # `profile delete` also removes the ~/.local/bin/<profile> wrapper script.
    _run(f"hermes profile delete {HERMES_PROFILE} --yes")
    print(f"✅ Profile [bold blue]{HERMES_PROFILE}[/bold blue] removed.")


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
    if os.path.exists(ZONES_CACHE_FILE):
        print(f"Zones cache file found at {ZONES_CACHE_FILE}")
    else:
        print(f"❌ Zones cache file not found at {ZONES_CACHE_FILE}")


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

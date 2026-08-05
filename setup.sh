#!/bin/sh
set -eu

# unattended-upgrades on a freshly booted TPU resolves ~230 packages and holds
# the dpkg lock for ~15 minutes, restarting sshd on its way through
# openssh-server. These are short-lived dev VMs, so take the auto-upgrade
# machinery out of the way rather than racing it.
sudo systemctl disable --now apt-daily.timer apt-daily-upgrade.timer \
    unattended-upgrades.service 2>/dev/null || true
sudo systemctl stop apt-daily.service apt-daily-upgrade.service 2>/dev/null || true

# Stopping the timer does not kill a run already in flight, so still wait for the
# lock instead of failing on it. NEEDRESTART_MODE=a keeps needrestart from
# prompting for which services to restart when there is no TTY to answer on.
APT="sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get -o DPkg::Lock::Timeout=900"

if dpkg -s python3-virtualenv python-is-python3 >/dev/null 2>&1; then
    echo "python3-virtualenv and python-is-python3 already present, skipping apt"
else
    $APT update
    $APT install -y python3-virtualenv python-is-python3
fi

git config --global credential.helper store

# Add user to docker group
sudo usermod -aG docker "$USER"

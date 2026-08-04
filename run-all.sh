#!/bin/sh
# Single remote entrypoint for a TPU install.
#
# get-tpu uploads one payload tarball and unpacks it into $HOME, then runs this
# script detached. Everything the install does happens inside this one process,
# so the install no longer depends on a series of separate ssh sessions staying
# up, and there is exactly one log to read when something goes wrong.
set -eu

cd ~

echo "=== setup.sh ==="
bash setup.sh

if [ -f extra/run.sh ]; then
    echo
    echo "=== extra startup ==="
    bash extra/run.sh
fi

echo
echo "=== install finished ==="

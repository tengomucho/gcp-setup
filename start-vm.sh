#!/bin/bash
usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }

# Parse arguments with getopt
[ $# -eq 0 ] && usage
while getopts ":hz:p:t:v:a:" arg; do
  case $arg in
    z) # Specify zone.
      zone=${OPTARG}
      ;;
    p) # Specify project name.
      project=${OPTARG}
      ;;
    t) # Specify TPU name.
      tpu_name=${OPTARG}
      ;;
    v) # Specify version. (default: v2-alpha-tpuv5-lite)
      version=${OPTARG}
      ;;
    a) # Specify accelerator type. (default: v5litepod-8)
      accelerator=${OPTARG}
      ;;
    h | *) # Display help.
      usage
      exit 0
      ;;
  esac
done

if [ -z "${zone}" ] || [ -z "${project}" ] || [ -z "${tpu_name}" ]; then
    usage
fi
if [ -z "${version}" ]; then
    version="v2-alpha-tpuv5-lite"
fi
if [ -z "${accelerator}" ]; then
    accelerator="v5litepod-8"
fi

# First check if VM is already up
echo "🔭 Checking if VM is already up..."
gcloud compute tpus tpu-vm list --zone=$zone|grep $tpu_name
if [ $? -eq 0 ]
then
    echo "❌ VM already up, quitting."
    exit 1
else
    echo "✅ Ready to create VM..."
fi

set -e

gcloud alpha compute tpus tpu-vm create $tpu_name \
--zone=$zone \
--accelerator-type=$accelerator \
--version v2-alpha-tpuv5-lite

echo "🧾 Copying setup script"
gcloud compute tpus tpu-vm scp --zone $zone setup.sh $tpu_name: \
    --project $project

echo "🤖 Retrieving IP and updating local settings"
EXT_IP=`gcloud compute tpus tpu-vm describe --zone=$zone $tpu_name |grep externalIp|cut -d ":" -f 2|cut -d ' ' -f 2`


# update .ssh/config
sed -i '' -E "/^Host $tpu_name$/,+1 s/(HostName ).*/\1$EXT_IP/" ~/.ssh/config

# Remove previous known host using the same IP
sed -i '' "/$EXT_IP/d" ~/.ssh/known_hosts
# This is not great, because it bypasses ssh security check, but that's ok
ssh-keyscan -H $EXT_IP >> ~/.ssh/known_hosts

echo "🏃 Running install script"
ssh $tpu_name -C "bash setup.sh"

echo
echo "✨ All ready. You can ssh $tpu_name now."
echo

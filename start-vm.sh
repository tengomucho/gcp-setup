#!/bin/bash
CONF=~/.gcp-setup

if test -f "$CONF"; then
    source $CONF
fi

if [ -z "$TPU_NAME" ]
then
    echo -n "Name of the VM: "
    read TPU_NAME
    echo "TPU_NAME=$TPU_NAME" >> $CONF
fi
if [ -z "$ZONE" ]
then
    echo -n "Zone: "
    read ZONE
    echo "ZONE=$ZONE" >> $CONF
fi
if [ -z "$PROJECT" ]
then
    echo -n "Project: "
    read PROJECT
    echo "PROJECT=$PROJECT" >> $CONF
fi

# First check if VM is already up
echo "🔭 Checking if VM is already up..."
gcloud compute tpus tpu-vm list --zone=$ZONE|grep $TPU_NAME
if [ $? -eq 0 ]
then
    echo "❌ VM already up, quitting."
    exit 1
else
    echo "✅ Ready to create VM..."
fi

gcloud alpha compute tpus tpu-vm create $TPU_NAME \
--zone=$ZONE \
--accelerator-type=v5litepod-8 \
--version v2-alpha-tpuv5-lite

echo "🧾 Copying setup script"
gcloud compute tpus tpu-vm scp --zone $ZONE setup.sh $TPU_NAME: \
    --project $PROJECT

echo "🤖 Retrieving IP and updating local settings"
EXT_IP=`gcloud compute tpus tpu-vm describe --zone=$ZONE $TPU_NAME |grep externalIp|cut -d ":" -f 2|cut -d ' ' -f 2`


# update .ssh/config
sed -i '' -E "/^Host $TPU_NAME$/,+1 s/(HostName ).*/\1$EXT_IP/" ~/.ssh/config

# Remove previous known host using the same IP
sed -i '' "/$EXT_IP/d" ~/.ssh/known_hosts

echo "🏃 Running install script"
ssh $TPU_NAME -C "bash setup.sh"

echo
echo "✨ All ready. You can ssh $TPU_NAME now."
echo

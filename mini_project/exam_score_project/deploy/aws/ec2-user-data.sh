#!/bin/bash
# deploy/aws/ec2-user-data.sh
#
# EC2 "User data" script -- paste this into the EC2 launch wizard's Advanced details ->
# User data field (or run it by hand over SSH; see the README's manual walkthrough,
# which is what actually TEACHES what each line below does). Either way, it runs
# automatically as root the first time the instance boots, and does everything needed to
# go from a bare Ubuntu EC2 instance to a running container: install Docker, pull the
# image from Docker Hub, run it.
#
# BEFORE USING: replace <dockerhub-username> below with your actual Docker Hub username
# (the same one you pushed exam-score-app to).
#
# Tested against: Ubuntu 22.04 LTS AMI, t2.micro / t3.micro (free-tier eligible).

set -euo pipefail

DOCKERHUB_USERNAME="<dockerhub-username>"
IMAGE="${DOCKERHUB_USERNAME}/exam-score-app:1.0"
CONTAINER_NAME="exam-score-app"

echo "[user-data] updating packages..."
apt-get update -y

echo "[user-data] installing Docker..."
apt-get install -y docker.io
systemctl enable docker
systemctl start docker

# Let the default `ubuntu` user run docker without sudo (takes effect on next login --
# not needed for this script itself, since it runs as root already).
usermod -aG docker ubuntu || true

echo "[user-data] pulling ${IMAGE} ..."
docker pull "${IMAGE}"

echo "[user-data] starting the container..."
# --restart unless-stopped: survives an instance reboot without you having to SSH back in
# and re-run `docker run` by hand -- a real, if small, production consideration.
docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    -p 8000:8000 \
    -p 8501:8501 \
    "${IMAGE}"

echo "[user-data] done. API: http://<this-instance-public-ip>:8000/docs  UI: http://<this-instance-public-ip>:8501"

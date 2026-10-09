#!/bin/bash
# ec2-user-data.sh: EC2 user data for Ubuntu Server 24.04 (amd64). ec2-launch.sh passes it to
# run-instances after setting MKTSTATS_IMAGE and MKTSTATS_REF below. EC2 runs it once, as root,
# at first boot; its output goes to /var/log/cloud-init-output.log.
#
# Documented, not run: the workshop authors have not run this script on EC2.
#
# It installs Docker Engine from Docker's apt repository, clones the workshop repository, writes
# a random JupyterLab token, and runs the workshop image as the systemd service mktstats-jupyter.
# JupyterLab is published on the instance's 127.0.0.1:8888 only (docker run -p 127.0.0.1:8888:8888):
# nothing listens on a public interface, so you reach it through an SSH tunnel or SSM port
# forwarding. Inside the container JupyterLab listens on the container's own interface.
#
# On the instance:
#   sudo cat /etc/mktstats/jupyter-token       the token (a new one is written only at first boot)
#   systemctl status mktstats-jupyter          service state; the first start pulls a multi-GB image
#   journalctl -u mktstats-jupyter -f          pull progress and JupyterLab log
#   /opt/mktstats/workshop                     the clone, mounted at /home/sagemaker-user/work
#   /opt/mktstats/cache                        MKTSTATS_CACHE, mounted at /home/sagemaker-user/cache
# If the image is private on GHCR the pull fails ("denied"/"unauthorized"); see README.md, EC2.
#
# Docs checked on 2026-10-09:
#   https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html (root, first boot, log file)
#   https://docs.docker.com/engine/install/ubuntu/ (apt repository steps, copied below)
#   https://docs.docker.com/reference/cli/docker/container/run/ (-p 127.0.0.1:..., --platform, --rm)
#   https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility
#   Jupyter Server reads the token from the file named by JUPYTER_TOKEN_FILE (checked in
#   jupyter_server/auth/identity.py of jupyter_server 2.20.0 inside the workshop image).
set -euo pipefail

# Set by ec2-launch.sh.
MKTSTATS_IMAGE="ghcr.io/project-delphi/marketing-statistics-workshop/env:latest"
MKTSTATS_REF="main"

REPO_URL="https://github.com/project-delphi/marketing-statistics-workshop.git"
export DEBIAN_FRONTEND=noninteractive

echo "== Docker Engine (Docker's apt repository)"
apt-get update
apt-get install -y ca-certificates curl git
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
# shellcheck disable=SC1091  # /etc/os-release exists on Ubuntu
cat > /etc/apt/sources.list.d/docker.sources << EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
systemctl enable --now docker

echo "== Workshop repository and cache (owned by the image's user, UID 1000 / GID 100)"
install -d -m 0755 /opt/mktstats
if [ ! -d /opt/mktstats/workshop/.git ]; then
  git clone --depth 1 --branch "$MKTSTATS_REF" "$REPO_URL" /opt/mktstats/workshop
fi
install -d /opt/mktstats/cache
chown -R 1000:100 /opt/mktstats/workshop /opt/mktstats/cache

echo "== JupyterLab token"
install -d -m 0755 /etc/mktstats
if [ ! -s /etc/mktstats/jupyter-token ]; then
  # 24 random bytes as hex; head reads a fixed amount, so nothing in the pipe gets SIGPIPE.
  head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > /etc/mktstats/jupyter-token
fi
# Readable by root (sudo cat) and by UID 1000, the user inside the container.
chown 1000:100 /etc/mktstats/jupyter-token
chmod 0400 /etc/mktstats/jupyter-token

echo "== systemd service mktstats-jupyter"
cat > /etc/systemd/system/mktstats-jupyter.service << EOF
[Unit]
Description=Statistics for Marketing workshop: JupyterLab in the workshop image
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Restart=on-failure
RestartSec=30
# The first pull is several GB.
TimeoutStartSec=0
ExecStartPre=-/usr/bin/docker rm -f mktstats-jupyter
ExecStartPre=/usr/bin/docker pull --platform linux/amd64 ${MKTSTATS_IMAGE}
ExecStart=/usr/bin/docker run --rm --name mktstats-jupyter --platform linux/amd64 \\
  -p 127.0.0.1:8888:8888 \\
  -v /opt/mktstats/workshop:/home/sagemaker-user/work \\
  -v /opt/mktstats/cache:/home/sagemaker-user/cache \\
  -v /etc/mktstats/jupyter-token:/run/mktstats/jupyter-token:ro \\
  -e JUPYTER_TOKEN_FILE=/run/mktstats/jupyter-token \\
  -e MKTSTATS_REPO_ROOT=/home/sagemaker-user/work \\
  -e PYTHONPATH=/home/sagemaker-user/work/src \\
  -e MKTSTATS_CACHE=/home/sagemaker-user/cache \\
  ${MKTSTATS_IMAGE} \\
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.root_dir=/home/sagemaker-user
ExecStop=/usr/bin/docker stop mktstats-jupyter

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
# --no-block: let cloud-init finish while the image is pulled.
systemctl enable --now --no-block mktstats-jupyter.service
echo "== Done. Token: sudo cat /etc/mktstats/jupyter-token; progress: journalctl -u mktstats-jupyter -f"

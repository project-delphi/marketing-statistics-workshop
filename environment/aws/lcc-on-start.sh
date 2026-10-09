#!/bin/bash
# lcc-on-start.sh: SageMaker Studio lifecycle configuration for JupyterLab spaces that use the
# default SageMaker Distribution image (the route without a custom image). Studio runs it each
# time the space starts. sagemaker-lifecycle-config.sh uploads it and sets MKTSTATS_REF below.
#
# Documented, not run: the workshop authors have not run this script on SageMaker.
#
# What it does (AWS stops a lifecycle script that runs longer than 5 minutes, so the slow part
# runs in the background with nohup):
#   1. clones the workshop repository into ~/marketing-statistics-workshop if it is not there
#      (an existing clone is left alone: it may hold your work);
#   2. starts a background pip install of the Python packages the labs need, at the versions
#      pinned in the clone's _variables.yml ("packages"), with environment/requirements.txt as
#      constraints, logging to ~/.mktstats-lcc.log.
# It installs nothing for R: the R labs are not supported on this route.
# Script output: CloudWatch log group /aws/sagemaker/studio, log stream
# <domain-id>/<space-name>/JupyterLab/default/LifecycleConfigOnStart.
#
# Docs checked on 2026-10-09:
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lifecycle-configurations.html
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lifecycle-configurations-create.html
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lifecycle-configurations-debug.html
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lifecycle-configurations-debug-timeout.html
set -eu

MKTSTATS_REF="main"
REPO_URL="https://github.com/project-delphi/marketing-statistics-workshop.git"
REPO_DIR="$HOME/marketing-statistics-workshop"
LOG="$HOME/.mktstats-lcc.log"

echo "mktstats lifecycle configuration: ref $MKTSTATS_REF; python3 is $(command -v python3) ($(python3 --version 2>&1))"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --quiet --depth 1 --branch "$MKTSTATS_REF" "$REPO_URL" "$REPO_DIR"
else
  echo "$REPO_DIR exists; leaving it as it is."
fi

# Read "name==version" for every entry under "packages:" in _variables.yml, the single source
# of truth for the lab package pins.
PINS="$(python3 - "$REPO_DIR/_variables.yml" << 'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
block = re.search(r"^packages:\n((?:[ #].*\n|\n)*)", text, re.M)
pins = re.findall(r'^  ([A-Za-z0-9_.-]+): \{version: "([^"]+)"', block.group(1) if block else "", re.M)
print(" ".join(f"{name}=={version}" for name, version in pins))
PY
)"
if [ -z "$PINS" ]; then
  echo "No pinned packages found in $REPO_DIR/_variables.yml; nothing installed." >&2
  exit 1
fi

echo "Installing in the background: $PINS (log: $LOG)"
# PINS is a space-separated list of requirement specifiers, split on purpose.
# shellcheck disable=SC2086
nohup python3 -m pip install --quiet $PINS -c "$REPO_DIR/environment/requirements.txt" >> "$LOG" 2>&1 &
echo "Lifecycle configuration finished; the pip install continues in the background."

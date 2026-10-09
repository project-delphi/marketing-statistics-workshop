#!/usr/bin/env bash
# ec2-launch.sh: launch one EC2 instance (Ubuntu Server 24.04, x86_64) that runs the workshop
# image with JupyterLab on the instance's 127.0.0.1:8888 (ec2-user-data.sh). Nothing is exposed
# to the internet except SSH from one address (EC2_ACCESS=ssh, the default) or nothing at all
# (EC2_ACCESS=ssm, Session Manager port forwarding).
#
# Documented, not run: the workshop authors have not run this script against AWS.
#
# Usage:
#   AWS_REGION=eu-west-1 EC2_SSH_CIDR=203.0.113.7/32 environment/aws/ec2-launch.sh [--dry-run] [--yes]
#   AWS_REGION=eu-west-1 EC2_ACCESS=ssm EC2_INSTANCE_PROFILE=my-ssm-profile environment/aws/ec2-launch.sh
#
# Environment variables:
#   AWS_REGION            required.
#   EC2_ACCESS            ssh (default) or ssm.
#   EC2_SSH_CIDR          ssh: required. The only address allowed to connect, normally your
#                         public IP as a /32, e.g. 203.0.113.7/32.
#   EC2_INSTANCE_PROFILE  ssm: required. An instance profile whose role has the AWS managed policy
#                         AmazonSSMManagedInstanceCore. Optional for ssh.
#   EC2_INSTANCE_TYPE     default m7i.xlarge (4 vCPU, 16 GiB). Must be x86_64: the image is amd64.
#   EC2_VOLUME_GB         root volume in GiB (default 40; gp3, encrypted, deleted with the instance)
#   EC2_IMAGE             image to run (default ghcr.io/project-delphi/marketing-statistics-workshop/env:latest;
#                         prefer a fixed :<hash> tag; the GHCR package may still be private)
#   MKTSTATS_REF          branch or tag of the workshop repository to clone (default: main)
#   EC2_KEY_NAME          ssh: key pair name (default mktstats-workshop)
#   EC2_KEY_FILE          ssh: where the new private key is saved (default ~/.ssh/mktstats-workshop.pem)
#   EC2_SG_NAME           security group name (default mktstats-workshop)
#   EC2_VPC_ID            default: the Region's default VPC
#   EC2_SUBNET_ID         optional subnet; it must give the instance a route to the internet
#                         (to install Docker and pull the image, and for SSH or SSM)
#   PROJECT_TAG_VALUE     value of the Project tag on everything created (default: mktstats-workshop)
#
# Needs: aws (CLI v2); for ssm also the Session Manager plugin for the AWS CLI.
#
# Docs checked on 2026-10-09:
#   https://documentation.ubuntu.com/aws/en/latest/aws-how-to/instances/find-ubuntu-images/
#     (SSM parameter /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id)
#   https://docs.aws.amazon.com/cli/latest/reference/ssm/get-parameter.html
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-images.html (RootDeviceName)
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-vpcs.html (filter is-default)
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-security-groups.html
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/create-security-group.html
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/authorize-security-group-ingress.html
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-key-pairs.html
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/create-key-pair.html (ed25519, KeyMaterial)
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instances.html (tag:<key>, instance-state-name)
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html (--count, --user-data file://,
#     --tag-specifications, --block-device-mappings, --metadata-options, --iam-instance-profile)
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/wait/instance-running.html
#   https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-IMDS-new-instances.html
#   https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/managing-users.html (Ubuntu user: ubuntu)
#   https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-sessions-start.html
#     (AWS-StartPortForwardingSession, portNumber, localPortNumber)
#   https://docs.aws.amazon.com/systems-manager/latest/userguide/setup-instance-permissions.html
#   https://docs.aws.amazon.com/systems-manager/latest/userguide/ami-preinstalled-agent.html
#     (SSM Agent is preinstalled on Ubuntu Server 24.04 LTS AMIs)
#   https://aws.amazon.com/ec2/instance-types/m7i/ (m7i.xlarge: 4 vCPU, 16 GiB)

set -euo pipefail
# shellcheck source=_lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

for arg in "$@"; do
  common_flag "$arg" || die "Unknown argument: $arg (see --help)"
done
require_region
banner
need_cmd aws

EC2_ACCESS="${EC2_ACCESS:-ssh}"
EC2_SSH_CIDR="${EC2_SSH_CIDR:-}"
EC2_INSTANCE_PROFILE="${EC2_INSTANCE_PROFILE:-}"
EC2_INSTANCE_TYPE="${EC2_INSTANCE_TYPE:-m7i.xlarge}"
EC2_VOLUME_GB="${EC2_VOLUME_GB:-40}"
EC2_IMAGE="${EC2_IMAGE:-ghcr.io/project-delphi/marketing-statistics-workshop/env:latest}"
MKTSTATS_REF="${MKTSTATS_REF:-main}"
EC2_KEY_NAME="${EC2_KEY_NAME:-mktstats-workshop}"
EC2_KEY_FILE="${EC2_KEY_FILE:-$HOME/.ssh/mktstats-workshop.pem}"
EC2_SG_NAME="${EC2_SG_NAME:-mktstats-workshop}"
EC2_VPC_ID="${EC2_VPC_ID:-}"
EC2_SUBNET_ID="${EC2_SUBNET_ID:-}"
AMI_PARAMETER="/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"

case "$EC2_ACCESS" in
  ssh)
    [[ -n "$EC2_SSH_CIDR" ]] || die "EC2_SSH_CIDR is required for EC2_ACCESS=ssh (your public IP as a /32)."
    [[ "$EC2_SSH_CIDR" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}/[0-9]{1,2}$ ]] || die "EC2_SSH_CIDR must look like 203.0.113.7/32"
    [[ "$EC2_SSH_CIDR" != 0.0.0.0/0 ]] || die "Refusing to open SSH to the whole internet (0.0.0.0/0)."
    ;;
  ssm)
    [[ -n "$EC2_INSTANCE_PROFILE" ]] ||
      die "EC2_INSTANCE_PROFILE is required for EC2_ACCESS=ssm (a role with AmazonSSMManagedInstanceCore)."
    ;;
  *) die "EC2_ACCESS must be ssh or ssm, not '$EC2_ACCESS'." ;;
esac
if ! [[ "$EC2_VOLUME_GB" =~ ^[0-9]+$ ]] || ((EC2_VOLUME_GB < 20)); then
  die "EC2_VOLUME_GB must be a number >= 20 (the image alone is several GB)."
fi
# These two values are written into the user-data script; keep them free of shell syntax.
[[ "$EC2_IMAGE" =~ ^[A-Za-z0-9./:@_-]+$ ]] || die "EC2_IMAGE contains unexpected characters: $EC2_IMAGE"
[[ "$MKTSTATS_REF" =~ ^[A-Za-z0-9._/-]+$ ]] || die "MKTSTATS_REF must be a branch or tag name: $MKTSTATS_REF"

TAG="{Key=$PROJECT_TAG_KEY,Value=$PROJECT_TAG_VALUE}"
make_tmpdir

note ""
note "== 1. Instances already tagged $PROJECT_TAG_KEY=$PROJECT_TAG_VALUE"
EXISTING="$(aws_capture "" ec2 describe-instances \
  --filters "Name=tag:$PROJECT_TAG_KEY,Values=$PROJECT_TAG_VALUE" \
  "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)"
if [[ -n "$EXISTING" && "$EXISTING" != None ]]; then
  note "Already running or stopped: $EXISTING"
  confirm "Launch another instance anyway?" || die "Stopped. Use the existing instance or run teardown.sh --ec2."
fi

note ""
note "== 2. Ubuntu Server 24.04 AMI (amd64) from Canonical's public SSM parameter"
AMI_ID="$(aws_capture "<ami-id>" ssm get-parameter --name "$AMI_PARAMETER" --query Parameter.Value --output text)"
ROOT_DEVICE="$(aws_capture "/dev/sda1" ec2 describe-images --image-ids "$AMI_ID" \
  --query 'Images[0].RootDeviceName' --output text)"
note "AMI $AMI_ID, root device $ROOT_DEVICE"

note ""
note "== 3. Network"
if [[ -z "$EC2_VPC_ID" ]]; then
  EC2_VPC_ID="$(aws_capture "<default-vpc-id>" ec2 describe-vpcs --filters Name=is-default,Values=true \
    --query 'Vpcs[0].VpcId' --output text)"
  [[ -n "$EC2_VPC_ID" && "$EC2_VPC_ID" != None ]] || die "No default VPC in $AWS_REGION; set EC2_VPC_ID and EC2_SUBNET_ID."
fi
SG_ID="$(aws_capture "None" ec2 describe-security-groups \
  --filters "Name=group-name,Values=$EC2_SG_NAME" "Name=vpc-id,Values=$EC2_VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text)"
if [[ -n "$SG_ID" && "$SG_ID" != None ]]; then
  note "Security group $EC2_SG_NAME ($SG_ID) exists; its rules are not changed."
else
  if [[ "$EC2_ACCESS" == ssh ]]; then
    SG_DESCRIPTION="Statistics for Marketing workshop: SSH from one address only"
  else
    SG_DESCRIPTION="Statistics for Marketing workshop: no inbound rules (SSM)"
  fi
  SG_ID="$(aws_capture "<sg-id>" ec2 create-security-group --group-name "$EC2_SG_NAME" \
    --description "$SG_DESCRIPTION" --vpc-id "$EC2_VPC_ID" \
    --tag-specifications "ResourceType=security-group,Tags=[$TAG]" \
    --query GroupId --output text)"
  if [[ "$EC2_ACCESS" == ssh ]]; then
    aws_run ec2 authorize-security-group-ingress --group-id "$SG_ID" \
      --protocol tcp --port 22 --cidr "$EC2_SSH_CIDR"
  fi
fi

KEY_ARGS=()
if [[ "$EC2_ACCESS" == ssh ]]; then
  note ""
  note "== 4. SSH key pair $EC2_KEY_NAME"
  if aws_probe ec2 describe-key-pairs --key-names "$EC2_KEY_NAME"; then
    [[ -f "$EC2_KEY_FILE" || "$DRY_RUN" == 1 ]] ||
      die "Key pair $EC2_KEY_NAME exists in AWS but $EC2_KEY_FILE does not; set EC2_KEY_FILE, or delete the key pair (teardown.sh --ec2)."
    note "Key pair exists; using $EC2_KEY_FILE."
  else
    [[ ! -e "$EC2_KEY_FILE" ]] || die "$EC2_KEY_FILE already exists; not overwriting it. Set EC2_KEY_FILE to a new path."
    show aws ec2 create-key-pair --key-name "$EC2_KEY_NAME" --key-type ed25519 \
      --tag-specifications "ResourceType=key-pair,Tags=[$TAG]" \
      --query KeyMaterial --output text --region "$AWS_REGION"
    note "  (output saved to $EC2_KEY_FILE with mode 600)"
    if [[ "$DRY_RUN" != 1 ]]; then
      mkdir -p "$(dirname "$EC2_KEY_FILE")"
      (
        umask 077
        aws ec2 create-key-pair --key-name "$EC2_KEY_NAME" --key-type ed25519 \
          --tag-specifications "ResourceType=key-pair,Tags=[$TAG]" \
          --query KeyMaterial --output text --region "$AWS_REGION" > "$EC2_KEY_FILE"
      )
      chmod 600 "$EC2_KEY_FILE"
    fi
  fi
  KEY_ARGS=(--key-name "$EC2_KEY_NAME")
fi

note ""
note "== 5. Launch"
USER_DATA="$TMP_DIR/user-data.sh"
sed -e "s|^MKTSTATS_IMAGE=.*|MKTSTATS_IMAGE=\"$EC2_IMAGE\"|" \
  -e "s|^MKTSTATS_REF=.*|MKTSTATS_REF=\"$MKTSTATS_REF\"|" \
  "$AWS_DIR/ec2-user-data.sh" > "$USER_DATA"
note "User data: ec2-user-data.sh with MKTSTATS_IMAGE=\"$EC2_IMAGE\" and MKTSTATS_REF=\"$MKTSTATS_REF\""
OPTIONAL_ARGS=()
if [[ -n "$EC2_SUBNET_ID" ]]; then OPTIONAL_ARGS+=(--subnet-id "$EC2_SUBNET_ID"); fi
if [[ -n "$EC2_INSTANCE_PROFILE" ]]; then OPTIONAL_ARGS+=(--iam-instance-profile "Name=$EC2_INSTANCE_PROFILE"); fi
BLOCK_DEVICES="[{\"DeviceName\":\"$ROOT_DEVICE\",\"Ebs\":{\"VolumeSize\":$EC2_VOLUME_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true,\"Encrypted\":true}}]"
note "An $EC2_INSTANCE_TYPE instance and its ${EC2_VOLUME_GB} GiB volume are billed until terminated."
confirm "Launch it now?" || die "Not launched."
# IMDSv2 only, hop limit 1: processes inside the Docker container cannot get a metadata token,
# so the notebooks cannot read the instance role's credentials (AWS suggests 2 only when
# containers need instance metadata).
INSTANCE_ID="$(aws_capture "<instance-id>" ec2 run-instances \
  --image-id "$AMI_ID" --instance-type "$EC2_INSTANCE_TYPE" --count 1 \
  --security-group-ids "$SG_ID" \
  ${KEY_ARGS[@]+"${KEY_ARGS[@]}"} ${OPTIONAL_ARGS[@]+"${OPTIONAL_ARGS[@]}"} \
  --block-device-mappings "$BLOCK_DEVICES" \
  --metadata-options "HttpEndpoint=enabled,HttpTokens=required,HttpPutResponseHopLimit=1" \
  --tag-specifications "ResourceType=instance,Tags=[$TAG,{Key=Name,Value=mktstats-workshop}]" \
  "ResourceType=volume,Tags=[$TAG]" \
  --user-data "file://$USER_DATA" \
  --query 'Instances[0].InstanceId' --output text)"
aws_run ec2 wait instance-running --instance-ids "$INSTANCE_ID"
PUBLIC_IP="$(aws_capture "<public-ip>" ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"

note ""
note "Launched $INSTANCE_ID ($PUBLIC_IP). First boot installs Docker and pulls the image (several GB;"
note "how long this takes on EC2 has not been measured). Then:"
if [[ "$EC2_ACCESS" == ssh ]]; then
  note "  ssh -i $EC2_KEY_FILE ubuntu@$PUBLIC_IP 'journalctl -u mktstats-jupyter -n 20'     # progress"
  note "  ssh -i $EC2_KEY_FILE ubuntu@$PUBLIC_IP 'sudo cat /etc/mktstats/jupyter-token'     # token"
  note "  ssh -i $EC2_KEY_FILE -N -L 8888:127.0.0.1:8888 ubuntu@$PUBLIC_IP                  # tunnel; leave it open"
else
  note "  aws ssm start-session --target $INSTANCE_ID --region $AWS_REGION"
  note "    then: sudo cat /etc/mktstats/jupyter-token   (and journalctl -u mktstats-jupyter -n 20)"
  note "  aws ssm start-session --target $INSTANCE_ID --region $AWS_REGION \\"
  note "    --document-name AWS-StartPortForwardingSession \\"
  note "    --parameters '{\"portNumber\":[\"8888\"],\"localPortNumber\":[\"8888\"]}'   # leave it open"
fi
note "  Open http://127.0.0.1:8888/lab?token=<token> in your browser."
note "Terminate it when you finish: AWS_REGION=$AWS_REGION environment/aws/teardown.sh --ec2"

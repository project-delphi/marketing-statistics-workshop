#!/usr/bin/env bash
# teardown.sh: delete what the scripts in environment/aws/ created, then list what is left.
# Nothing is deleted unless you name it with a flag, and every deletion asks first (--yes skips
# the questions). Each step skips what is already gone, so it is safe to run again.
#
# Documented, not run: the workshop authors have not run this script against AWS.
#
# Usage:
#   AWS_REGION=eu-west-1 environment/aws/teardown.sh --ec2 [--dry-run] [--yes]
#   AWS_REGION=eu-west-1 SM_DOMAIN_ID=d-xxxxxxxxxxxx SM_SPACE_NAME=mktstats-alice \
#     environment/aws/teardown.sh --sagemaker-apps --sagemaker-image --lcc
#   AWS_REGION=eu-west-1 S3_BUCKET=my-bucket environment/aws/teardown.sh --ecr-repo --s3-bucket
#
# Flags (run in this order, whatever order you give them):
#   --ec2              terminate every instance tagged Project=$PROJECT_TAG_VALUE (their root volumes
#                      go with them), then delete the security group EC2_SG_NAME and the key pair
#                      EC2_KEY_NAME if they carry that tag. The local private key file is kept.
#   --sagemaker-apps   delete the JupyterLab apps that run the workshop image, then the space
#                      SM_SPACE_NAME if set (deleting a space deletes the files stored in it).
#   --lcc              remove the lifecycle configuration SM_LCC_NAME from the domain and delete it.
#   --sagemaker-image  remove the image from the domain's custom images, delete the app image config,
#                      the image versions and the image (the ECR copy stays; see --ecr-repo).
#   --ecr-repo         delete the ECR repository ECR_REPO and every image in it.
#   --s3-bucket        delete the bucket S3_BUCKET and every object in it.
#   --dry-run, --yes, --help
#
# Environment variables (defaults match the other scripts):
#   AWS_REGION (required), SM_DOMAIN_ID (required for --sagemaker-*, --lcc), SM_IMAGE_NAME
#   (mktstats-env), SM_APP_IMAGE_CONFIG (mktstats-env-jupyterlab), SM_SPACE_NAME, SM_LCC_NAME
#   (mktstats-jupyterlab-on-start), ECR_REPO (mktstats-env), S3_BUCKET (required for --s3-bucket),
#   EC2_SG_NAME (mktstats-workshop), EC2_KEY_NAME (mktstats-workshop), EC2_KEY_FILE
#   (~/.ssh/mktstats-workshop.pem), PROJECT_TAG_VALUE (mktstats-workshop)
#
# Docs checked on 2026-10-09:
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-byoi-how-to-detach-from-domain.html
#     (detach before deleting the image; delete every app in the domain before updating custom
#     images; delete-app-image-config; delete-image also deletes the image versions)
#   https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-delete-domain.html (list-apps, delete-app,
#     list-spaces, delete-space; deleted spaces lose their data)
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instances.html
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/terminate-instances.html
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/wait/instance-terminated.html
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-security-groups.html
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-security-group.html
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-key-pairs.html (filters key-name, tag:<key>)
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/delete-key-pair.html
#   https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-volumes.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/list-apps.html (ResourceSpec.SageMakerImageArn)
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/delete-app.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-app.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-space.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/delete-space.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/list-spaces.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-domain.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/update-domain.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-studio-lifecycle-config.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/delete-studio-lifecycle-config.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/list-studio-lifecycle-configs.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-app-image-config.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/delete-app-image-config.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/list-app-image-configs.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/list-image-versions.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/delete-image-version.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-image.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/delete-image.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/list-images.html
#   https://docs.aws.amazon.com/cli/latest/reference/ecr/describe-repositories.html
#   https://docs.aws.amazon.com/cli/latest/reference/ecr/delete-repository.html (--force)
#   https://docs.aws.amazon.com/cli/latest/reference/s3api/head-bucket.html
#   https://docs.aws.amazon.com/cli/latest/reference/s3/rb.html (--force; versioned objects are not removed)
#   https://docs.aws.amazon.com/cli/latest/reference/resourcegroupstaggingapi/get-resources.html

set -euo pipefail
# shellcheck source=_lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

DO_EC2=0 DO_APPS=0 DO_LCC=0 DO_IMAGE=0 DO_ECR=0 DO_S3=0
for arg in "$@"; do
  case "$arg" in
    --ec2) DO_EC2=1 ;;
    --sagemaker-apps) DO_APPS=1 ;;
    --lcc) DO_LCC=1 ;;
    --sagemaker-image) DO_IMAGE=1 ;;
    --ecr-repo) DO_ECR=1 ;;
    --s3-bucket) DO_S3=1 ;;
    *) common_flag "$arg" || die "Unknown argument: $arg (see --help)" ;;
  esac
done
require_region
if ((DO_EC2 + DO_APPS + DO_LCC + DO_IMAGE + DO_ECR + DO_S3 == 0)); then
  usage
  die "Nothing selected: name what to delete with a flag (nothing is deleted by default)."
fi
# In a dry run, show the delete path for everything.
DRY_RUN_ASSUME_EXISTS=1
banner
need_cmd aws

SM_DOMAIN_ID="${SM_DOMAIN_ID:-}"
SM_IMAGE_NAME="${SM_IMAGE_NAME:-mktstats-env}"
SM_APP_IMAGE_CONFIG="${SM_APP_IMAGE_CONFIG:-mktstats-env-jupyterlab}"
SM_SPACE_NAME="${SM_SPACE_NAME:-}"
SM_LCC_NAME="${SM_LCC_NAME:-mktstats-jupyterlab-on-start}"
ECR_REPO="${ECR_REPO:-mktstats-env}"
S3_BUCKET="${S3_BUCKET:-}"
EC2_SG_NAME="${EC2_SG_NAME:-mktstats-workshop}"
EC2_KEY_NAME="${EC2_KEY_NAME:-mktstats-workshop}"
EC2_KEY_FILE="${EC2_KEY_FILE:-$HOME/.ssh/mktstats-workshop.pem}"
TAG_FILTER="Name=tag:$PROJECT_TAG_KEY,Values=$PROJECT_TAG_VALUE"

if ((DO_APPS + DO_LCC + DO_IMAGE > 0)); then
  require_var SM_DOMAIN_ID
  command -v jq > /dev/null 2>&1 || die "jq is required for the SageMaker steps (also for --dry-run)."
  make_tmpdir
fi
if ((DO_S3 == 1)); then require_var S3_BUCKET; fi

# update_jupyterlab_settings JQ_FILTER DESCRIPTION: read the domain's JupyterLab settings, apply
# JQ_FILTER, and send them back if anything changed.
update_jupyterlab_settings() {
  local filter=$1 what=$2 current new
  current="$(aws_capture "$3" sagemaker describe-domain --domain-id "$SM_DOMAIN_ID" \
    --query DefaultUserSettings.JupyterLabAppSettings --output json)"
  new="$(printf '%s' "$current" | jq -c "(. // {}) | $filter")"
  if [[ "$(printf '%s' "$current" | jq -c '. // {}')" == "$new" && "$DRY_RUN" != 1 ]]; then
    note "Domain $SM_DOMAIN_ID: nothing to change ($what)."
    return 0
  fi
  printf '%s' "$new" | jq '{JupyterLabAppSettings: .}' > "$TMP_DIR/default-user-settings.json"
  show_file "$TMP_DIR/default-user-settings.json"
  confirm "Update domain $SM_DOMAIN_ID: $what?" || {
    note "Skipped."
    return 1
  }
  aws_run sagemaker update-domain --domain-id "$SM_DOMAIN_ID" \
    --default-user-settings "file://$TMP_DIR/default-user-settings.json"
  WAIT_OK="InService" WAIT_FAIL="Failed Update_Failed Delete_Failed Deleting" \
    wait_for "Domain $SM_DOMAIN_ID" 900 15 \
    aws sagemaker describe-domain --domain-id "$SM_DOMAIN_ID" --query Status --output text --region "$AWS_REGION"
}

if ((DO_EC2 == 1)); then
  note ""
  note "== EC2: instances, security group, key pair (tag $PROJECT_TAG_KEY=$PROJECT_TAG_VALUE)"
  IDS="$(aws_capture "i-0123456789abcdef0" ec2 describe-instances --filters "$TAG_FILTER" \
    "Name=instance-state-name,Values=pending,running,shutting-down,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text)"
  read -r -a INSTANCE_IDS <<< "${IDS//None/}"
  if ((${#INSTANCE_IDS[@]} > 0)); then
    if confirm "Terminate ${INSTANCE_IDS[*]} (and their root volumes)?"; then
      aws_run ec2 terminate-instances --instance-ids "${INSTANCE_IDS[@]}"
      aws_run ec2 wait instance-terminated --instance-ids "${INSTANCE_IDS[@]}"
    fi
  else
    note "No tagged instances."
  fi
  SG_IDS="$(aws_capture "sg-0123456789abcdef0" ec2 describe-security-groups \
    --filters "Name=group-name,Values=$EC2_SG_NAME" "$TAG_FILTER" \
    --query 'SecurityGroups[].GroupId' --output text)"
  read -r -a SG_LIST <<< "${SG_IDS//None/}"
  for sg in ${SG_LIST[@]+"${SG_LIST[@]}"}; do
    if confirm "Delete security group $EC2_SG_NAME ($sg)?"; then
      aws_run ec2 delete-security-group --group-id "$sg"
    fi
  done
  ((${#SG_LIST[@]} > 0)) || note "No tagged security group named $EC2_SG_NAME."
  KEY="$(aws_capture "$EC2_KEY_NAME" ec2 describe-key-pairs \
    --filters "Name=key-name,Values=$EC2_KEY_NAME" "$TAG_FILTER" \
    --query 'KeyPairs[0].KeyName' --output text)"
  if [[ -n "$KEY" && "$KEY" != None ]]; then
    if confirm "Delete key pair $EC2_KEY_NAME?"; then
      aws_run ec2 delete-key-pair --key-name "$EC2_KEY_NAME"
      note "The private key file $EC2_KEY_FILE is still on this computer; delete it if you no longer need it."
    fi
  else
    note "No tagged key pair named $EC2_KEY_NAME."
  fi
fi

if ((DO_APPS + DO_IMAGE > 0)); then
  ACCOUNT_ID="$(aws_capture "<account-id>" sts get-caller-identity --query Account --output text)"
  IMAGE_ARN="arn:aws:sagemaker:$AWS_REGION:$ACCOUNT_ID:image/$SM_IMAGE_NAME"
fi

if ((DO_APPS == 1)); then
  note ""
  note "== SageMaker: JupyterLab apps running $SM_IMAGE_NAME in domain $SM_DOMAIN_ID"
  APPS="$(aws_capture "mktstats-alice None JupyterLab default" sagemaker list-apps --domain-id-equals "$SM_DOMAIN_ID" \
    --query "Apps[?ResourceSpec.SageMakerImageArn=='$IMAGE_ARN' && Status!='Deleted' && Status!='Deleting'].[SpaceName,UserProfileName,AppType,AppName]" \
    --output text)"
  if [[ -z "$APPS" ]]; then note "No apps run $SM_IMAGE_NAME."; fi
  # Read the list on file descriptor 3 so nothing inside the loop can consume it.
  while read -r space profile type name <&3; do
    [[ -n "${name:-}" ]] || continue
    if [[ "$space" != None ]]; then OWNER=(--space-name "$space"); else OWNER=(--user-profile-name "$profile"); fi
    if confirm "Delete app $type/$name in ${OWNER[1]} (unsaved work in open notebooks is lost)?"; then
      aws_run sagemaker delete-app --domain-id "$SM_DOMAIN_ID" "${OWNER[@]}" --app-type "$type" --app-name "$name"
      WAIT_OK="Deleted GONE" WAIT_FAIL="Failed" wait_for "App $type/$name in ${OWNER[1]}" 900 15 \
        aws sagemaker describe-app --domain-id "$SM_DOMAIN_ID" "${OWNER[@]}" --app-type "$type" \
        --app-name "$name" --query Status --output text --region "$AWS_REGION"
    fi
  done 3<<< "$APPS"
  if [[ -n "$SM_SPACE_NAME" ]]; then
    if aws_probe sagemaker describe-space --domain-id "$SM_DOMAIN_ID" --space-name "$SM_SPACE_NAME"; then
      if confirm "Delete space $SM_SPACE_NAME and every file stored in it?"; then
        aws_run sagemaker delete-space --domain-id "$SM_DOMAIN_ID" --space-name "$SM_SPACE_NAME"
        WAIT_OK="GONE" WAIT_FAIL="Delete_Failed" wait_for "Space $SM_SPACE_NAME" 900 15 \
          aws sagemaker describe-space --domain-id "$SM_DOMAIN_ID" --space-name "$SM_SPACE_NAME" \
          --query Status --output text --region "$AWS_REGION"
      fi
    else
      note "No space named $SM_SPACE_NAME."
    fi
  fi
fi

if ((DO_LCC == 1)); then
  note ""
  note "== SageMaker: lifecycle configuration $SM_LCC_NAME"
  LCC_ARN="$(aws_capture_quiet "arn:aws:sagemaker:$AWS_REGION:<account-id>:studio-lifecycle-config/$SM_LCC_NAME" \
    sagemaker describe-studio-lifecycle-config --studio-lifecycle-config-name "$SM_LCC_NAME" \
    --query StudioLifecycleConfigArn --output text || true)"
  if [[ -z "$LCC_ARN" || "$LCC_ARN" == None ]]; then
    note "No lifecycle configuration named $SM_LCC_NAME."
  else
    # The API pattern for DefaultResourceSpec.LifecycleConfigArn allows the value "None"; the docs
    # say to pass None to clear SageMakerImageVersionArn and we assume the same here (unverified).
    if update_jupyterlab_settings \
      ".LifecycleConfigArns = ((.LifecycleConfigArns // []) - [\"$LCC_ARN\"])
       | if .DefaultResourceSpec.LifecycleConfigArn == \"$LCC_ARN\" then .DefaultResourceSpec.LifecycleConfigArn = \"None\" else . end" \
      "remove lifecycle configuration $SM_LCC_NAME" \
      "{\"LifecycleConfigArns\": [\"$LCC_ARN\"]}"; then
      if confirm "Delete lifecycle configuration $SM_LCC_NAME?"; then
        aws_run sagemaker delete-studio-lifecycle-config --studio-lifecycle-config-name "$SM_LCC_NAME"
      fi
    fi
  fi
fi

if ((DO_IMAGE == 1)); then
  note ""
  note "== SageMaker: image $SM_IMAGE_NAME and app image config $SM_APP_IMAGE_CONFIG"
  note "AWS: detach the image before deleting it, and delete every app in the domain before the"
  note "custom-image list can be updated. If update-domain fails, stop the remaining apps and rerun."
  if update_jupyterlab_settings \
    ".CustomImages = [(.CustomImages // [])[] | select(.ImageName != \"$SM_IMAGE_NAME\")]" \
    "remove custom image $SM_IMAGE_NAME" \
    "{\"CustomImages\": [{\"ImageName\": \"$SM_IMAGE_NAME\", \"ImageVersionNumber\": 1, \"AppImageConfigName\": \"$SM_APP_IMAGE_CONFIG\"}]}"; then
    if aws_probe sagemaker describe-app-image-config --app-image-config-name "$SM_APP_IMAGE_CONFIG"; then
      if confirm "Delete app image config $SM_APP_IMAGE_CONFIG?"; then
        aws_run sagemaker delete-app-image-config --app-image-config-name "$SM_APP_IMAGE_CONFIG"
      fi
    else
      note "No app image config named $SM_APP_IMAGE_CONFIG."
    fi
    if aws_probe sagemaker describe-image --image-name "$SM_IMAGE_NAME"; then
      VERSIONS="$(aws_capture "1" sagemaker list-image-versions --image-name "$SM_IMAGE_NAME" \
        --query 'ImageVersions[].Version' --output text)"
      if confirm "Delete image $SM_IMAGE_NAME and its versions (${VERSIONS:-none})? The ECR image stays."; then
        for version in ${VERSIONS//None/}; do
          aws_run sagemaker delete-image-version --image-name "$SM_IMAGE_NAME" --version-number "$version"
        done
        WAIT_OK="0 GONE" wait_for "Image versions of $SM_IMAGE_NAME" 900 15 \
          aws sagemaker list-image-versions --image-name "$SM_IMAGE_NAME" \
          --query 'length(ImageVersions)' --output text --region "$AWS_REGION"
        # delete-image also deletes any versions that are left (AWS docs).
        aws_run sagemaker delete-image --image-name "$SM_IMAGE_NAME"
        WAIT_OK="GONE" WAIT_FAIL="DELETE_FAILED" wait_for "Image $SM_IMAGE_NAME" 900 15 \
          aws sagemaker describe-image --image-name "$SM_IMAGE_NAME" --query ImageStatus --output text \
          --region "$AWS_REGION"
      fi
    else
      note "No SageMaker image named $SM_IMAGE_NAME."
    fi
  else
    note "The image is still attached to the domain; not deleting it (AWS requires detaching first)."
  fi
fi

if ((DO_ECR == 1)); then
  note ""
  note "== ECR repository $ECR_REPO"
  if aws_probe ecr describe-repositories --repository-names "$ECR_REPO"; then
    if confirm "Delete ECR repository $ECR_REPO and every image in it?"; then
      aws_run ecr delete-repository --repository-name "$ECR_REPO" --force
    fi
  else
    note "No ECR repository named $ECR_REPO."
  fi
fi

if ((DO_S3 == 1)); then
  note ""
  note "== S3 bucket $S3_BUCKET"
  if aws_probe s3api head-bucket --bucket "$S3_BUCKET"; then
    # rb --force deletes every object, then the bucket. It does not delete old object versions,
    # so it fails on a versioned bucket; the README's bucket setup does not turn versioning on.
    if confirm "Delete bucket s3://$S3_BUCKET and every object in it?"; then
      aws_run s3 rb "s3://$S3_BUCKET" --force
    fi
  else
    note "No bucket named $S3_BUCKET (or no access to it)."
  fi
fi

note ""
note "== What is left (read-only checks; empty output or 'not found' errors mean nothing is left)"
check() {
  show aws "$@" --region "$AWS_REGION"
  if [[ "$DRY_RUN" != 1 ]]; then aws "$@" --region "$AWS_REGION" || true; fi
}
check resourcegroupstaggingapi get-resources --tag-filters "Key=$PROJECT_TAG_KEY,Values=$PROJECT_TAG_VALUE" \
  --query 'ResourceTagMappingList[].ResourceARN' --output text
check ec2 describe-instances --filters "$TAG_FILTER" \
  "Name=instance-state-name,Values=pending,running,shutting-down,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text
check ec2 describe-volumes --filters "$TAG_FILTER" --query 'Volumes[].VolumeId' --output text
check ec2 describe-security-groups --filters "$TAG_FILTER" --query 'SecurityGroups[].GroupId' --output text
check ec2 describe-key-pairs --filters "$TAG_FILTER" --query 'KeyPairs[].KeyName' --output text
if [[ -n "$SM_DOMAIN_ID" ]]; then
  check sagemaker list-apps --domain-id-equals "$SM_DOMAIN_ID" \
    --query "Apps[?Status!='Deleted'].[SpaceName,AppType,AppName,Status]" --output text
  check sagemaker list-spaces --domain-id-equals "$SM_DOMAIN_ID" --query 'Spaces[].SpaceName' --output text
fi
check sagemaker list-images --name-contains "$SM_IMAGE_NAME" --query 'Images[].ImageName' --output text
check sagemaker list-app-image-configs --name-contains "$SM_APP_IMAGE_CONFIG" \
  --query 'AppImageConfigs[].AppImageConfigName' --output text
check sagemaker list-studio-lifecycle-configs --name-contains "$SM_LCC_NAME" \
  --query 'StudioLifecycleConfigs[].StudioLifecycleConfigName' --output text
check ecr describe-repositories --repository-names "$ECR_REPO" --query 'repositories[].repositoryUri' --output text
if [[ -n "$S3_BUCKET" ]]; then check s3api head-bucket --bucket "$S3_BUCKET"; fi
note "Tagging covers what these scripts tagged; check the Billing console's cost breakdown too."

#!/usr/bin/env bash
# sagemaker-register-image.sh: make the workshop image (already pushed to ECR by build-push-ecr.sh)
# available as a custom JupyterLab image in the new Amazon SageMaker Studio, and optionally start
# a private JupyterLab space that runs it.
#
# Documented, not run: the workshop authors have not run this script against AWS.
#
# Steps: create-image -> create-image-version -> create-app-image-config (JupyterLab container
# settings) -> update-domain (add the image to the domain's JupyterLab custom images) ->
# optionally create-space + create-app.
#
# Usage:
#   AWS_REGION=eu-west-1 SM_DOMAIN_ID=d-xxxxxxxxxxxx \
#   SM_EXECUTION_ROLE_ARN=arn:aws:iam::111122223333:role/MySageMakerExecutionRole \
#   IMAGE_URI=111122223333.dkr.ecr.eu-west-1.amazonaws.com/mktstats-env:<tag> \
#   environment/aws/sagemaker-register-image.sh [--dry-run] [--help]
#
# Environment variables:
#   AWS_REGION             required.
#   SM_DOMAIN_ID           required. The Studio domain (aws sagemaker list-domains).
#   SM_EXECUTION_ROLE_ARN  required. Role passed to create-image; it must be allowed to pull the
#                          image from ECR (see iam/sagemaker-execution-role.json).
#   IMAGE_URI              the ECR image. Default: <account>.dkr.ecr.<region>.amazonaws.com/
#                          $ECR_REPO:$IMAGE_TAG (ECR_REPO defaults to mktstats-env; then IMAGE_TAG
#                          is required).
#   SM_IMAGE_NAME          SageMaker image name, lowercase (default: mktstats-env)
#   SM_APP_IMAGE_CONFIG    app image config name (default: mktstats-env-jupyterlab)
#   SM_REPO_DIR            where the workshop repository is cloned inside a space
#                          (default: /home/sagemaker-user/marketing-statistics-workshop)
#   SM_SET_REPO_ENV        1 (default): set MKTSTATS_REPO_ROOT=$SM_REPO_DIR,
#                          PYTHONPATH=$SM_REPO_DIR/src and MKTSTATS_CACHE in the container, so
#                          notebooks use the clone; 0: set no environment variables.
#   SM_USER_PROFILE        optional; together with SM_SPACE_NAME: create a private JupyterLab
#   SM_SPACE_NAME          space owned by this user profile, running this image, and start it.
#   SM_INSTANCE_TYPE       instance type of that space (default: ml.m5.xlarge, 4 vCPU / 16 GiB)
#   PROJECT_TAG_VALUE      value of the Project tag on created resources (default: mktstats-workshop)
#
# Needs: aws (CLI v2), jq (also for --dry-run: it builds the JSON payloads, which are printed).
# Each run that sees a new IMAGE_URI adds an image version; rerunning with the same IMAGE_URI
# reuses the latest version. An existing app image config or space is reused, not changed.
#
# Docs checked on 2026-10-09:
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html (Studio Classic: "no longer available
#     for onboarding", so only the new Studio is covered here)
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated.html (the new Studio)
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-byoi.html
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-byoi-specs.html (UID 1000, GID 100,
#     EBS volume mounted at /home/sagemaker-user)
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-admin-guide-custom-images.html
#     (base URL jupyterlab/default, health check jupyterlab/default/api/status on port 8888,
#     token/password off, all origins allowed)
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-byoi-how-to.html
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-byoi-how-to-container-configuration.html
#     (update-domain --default-user-settings JupyterLabAppSettings.CustomImages)
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-byoi-how-to-launch.html
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-byoi-how-to-detach-from-domain.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-image.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-image.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-image-version.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-image-version.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-app-image-config.html
#     (JupyterLabAppImageConfig: FileSystemConfig, ContainerConfig.ContainerEntrypoint (max 1 item),
#     ContainerArguments (max 50, each max 64 chars), ContainerEnvironmentVariables (max 25))
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-app-image-config.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-domain.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/update-domain.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-space.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-space.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-app.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-app.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-presigned-domain-url.html

set -euo pipefail
# shellcheck source=_lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

for arg in "$@"; do
  common_flag "$arg" || die "Unknown argument: $arg (see --help)"
done
require_region
banner

require_var SM_DOMAIN_ID
require_var SM_EXECUTION_ROLE_ARN
need_cmd aws
command -v jq > /dev/null 2>&1 || die "jq is required (also for --dry-run: it builds the JSON payloads)."

SM_IMAGE_NAME="${SM_IMAGE_NAME:-mktstats-env}"
SM_APP_IMAGE_CONFIG="${SM_APP_IMAGE_CONFIG:-mktstats-env-jupyterlab}"
SM_REPO_DIR="${SM_REPO_DIR:-/home/sagemaker-user/marketing-statistics-workshop}"
SM_SET_REPO_ENV="${SM_SET_REPO_ENV:-1}"
SM_INSTANCE_TYPE="${SM_INSTANCE_TYPE:-ml.m5.xlarge}"
SM_USER_PROFILE="${SM_USER_PROFILE:-}"
SM_SPACE_NAME="${SM_SPACE_NAME:-}"
ECR_REPO="${ECR_REPO:-mktstats-env}"

# Image-version ARNs only allow lowercase image names (create-space / update-domain patterns).
[[ "$SM_IMAGE_NAME" =~ ^[a-z0-9]([-.]?[a-z0-9]){0,62}$ ]] ||
  die "SM_IMAGE_NAME must be lowercase letters, digits, '-' or '.': $SM_IMAGE_NAME"
[[ "$SM_APP_IMAGE_CONFIG" =~ ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$ ]] ||
  die "SM_APP_IMAGE_CONFIG must be letters, digits and '-': $SM_APP_IMAGE_CONFIG"
[[ "$SM_REPO_DIR" == /home/sagemaker-user/* ]] ||
  die "SM_REPO_DIR must be under /home/sagemaker-user (the space's persistent volume)."
if [[ -n "$SM_USER_PROFILE" && -z "$SM_SPACE_NAME" ]] || [[ -z "$SM_USER_PROFILE" && -n "$SM_SPACE_NAME" ]]; then
  die "Set both SM_USER_PROFILE and SM_SPACE_NAME to create a space, or neither."
fi

ACCOUNT_ID="$(aws_capture "<account-id>" sts get-caller-identity --query Account --output text)"
if [[ -z "${IMAGE_URI:-}" ]]; then
  require_var IMAGE_TAG
  IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"
fi
note "Image: $IMAGE_URI"
make_tmpdir
TAGS="Key=$PROJECT_TAG_KEY,Value=$PROJECT_TAG_VALUE"

note ""
note "== 1. SageMaker image $SM_IMAGE_NAME"
if aws_probe sagemaker describe-image --image-name "$SM_IMAGE_NAME"; then
  note "Image $SM_IMAGE_NAME exists."
else
  aws_run sagemaker create-image --image-name "$SM_IMAGE_NAME" \
    --role-arn "$SM_EXECUTION_ROLE_ARN" \
    --display-name "Statistics for Marketing workshop" \
    --description "R 4.6.1 + Python 3.13 + JupyterLab (environment/Dockerfile)" \
    --tags "$TAGS"
fi
WAIT_OK="CREATED" WAIT_FAIL="CREATE_FAILED DELETING DELETE_FAILED" \
  WAIT_HINT="See FailureReason in: aws sagemaker describe-image --image-name $SM_IMAGE_NAME" \
  wait_for "Image $SM_IMAGE_NAME" 600 10 \
  aws sagemaker describe-image --image-name "$SM_IMAGE_NAME" --query ImageStatus --output text --region "$AWS_REGION"
IMAGE_ARN="$(aws_capture "arn:aws:sagemaker:$AWS_REGION:$ACCOUNT_ID:image/$SM_IMAGE_NAME" \
  sagemaker describe-image --image-name "$SM_IMAGE_NAME" --query ImageArn --output text)"

note ""
note "== 2. Image version for $IMAGE_URI"
LATEST_BASE="$(aws_capture_quiet "<none>" sagemaker describe-image-version --image-name "$SM_IMAGE_NAME" \
  --query BaseImage --output text || true)"
if [[ "$LATEST_BASE" == "$IMAGE_URI" ]]; then
  note "The latest version already uses $IMAGE_URI; reusing it."
  VERSION_ARN="$(aws_capture "arn:aws:sagemaker:$AWS_REGION:$ACCOUNT_ID:image-version/$SM_IMAGE_NAME/1" \
    sagemaker describe-image-version --image-name "$SM_IMAGE_NAME" --query ImageVersionArn --output text)"
else
  VERSION_ARN="$(aws_capture "arn:aws:sagemaker:$AWS_REGION:$ACCOUNT_ID:image-version/$SM_IMAGE_NAME/1" \
    sagemaker create-image-version --image-name "$SM_IMAGE_NAME" --base-image "$IMAGE_URI" \
    --query ImageVersionArn --output text)"
fi
VERSION="${VERSION_ARN##*/}"
[[ "$VERSION" =~ ^[0-9]+$ ]] || die "Could not read the version number from $VERSION_ARN"
WAIT_OK="CREATED" WAIT_FAIL="CREATE_FAILED DELETING DELETE_FAILED" \
  WAIT_HINT="See FailureReason in: aws sagemaker describe-image-version --image-name $SM_IMAGE_NAME --version-number $VERSION" \
  wait_for "Image version $VERSION" 1800 15 \
  aws sagemaker describe-image-version --image-name "$SM_IMAGE_NAME" --version-number "$VERSION" \
  --query ImageVersionStatus --output text --region "$AWS_REGION"

note ""
note "== 3. App image config $SM_APP_IMAGE_CONFIG (how Studio starts the container)"
# JupyterLab as SageMaker requires it: port 8888, base URL /jupyterlab/default (the health check is
# jupyterlab/default/api/status), no token or password (Studio authenticates), all origins allowed.
# The image has no ENTRYPOINT; ContainerEntrypoint takes at most one item, so the program is
# jupyter-lab and everything else is an argument.
if [[ "$SM_SET_REPO_ENV" == 1 ]]; then
  ENV_JSON="$(jq -n --arg root "$SM_REPO_DIR" \
    '{MKTSTATS_REPO_ROOT: $root, PYTHONPATH: ($root + "/src"), MKTSTATS_CACHE: "/home/sagemaker-user/.cache/mktstats"}')"
else
  ENV_JSON="{}"
fi
APP_IMAGE_CONFIG_JSON="$TMP_DIR/jupyter-lab-app-image-config.json"
jq -n --argjson env "$ENV_JSON" '{
  FileSystemConfig: {MountPath: "/home/sagemaker-user", DefaultUid: 1000, DefaultGid: 100},
  ContainerConfig: {
    ContainerEntrypoint: ["jupyter-lab"],
    ContainerArguments: [
      "--ip=0.0.0.0",
      "--port=8888",
      "--no-browser",
      "--ServerApp.base_url=/jupyterlab/default",
      "--ServerApp.token=",
      "--ServerApp.allow_origin=*",
      "--ServerApp.root_dir=/home/sagemaker-user"
    ],
    ContainerEnvironmentVariables: $env
  }
}' > "$APP_IMAGE_CONFIG_JSON"
if aws_probe sagemaker describe-app-image-config --app-image-config-name "$SM_APP_IMAGE_CONFIG"; then
  note "App image config $SM_APP_IMAGE_CONFIG exists; reusing it unchanged."
  note "(To change it: teardown.sh --sagemaker-image, then rerun this script.)"
else
  show_file "$APP_IMAGE_CONFIG_JSON"
  aws_run sagemaker create-app-image-config --app-image-config-name "$SM_APP_IMAGE_CONFIG" \
    --jupyter-lab-app-image-config "file://$APP_IMAGE_CONFIG_JSON" \
    --tags "$TAGS"
fi

note ""
note "== 4. Attach the image to domain $SM_DOMAIN_ID (JupyterLab custom images for private spaces)"
# update-domain replaces the list of custom images, so read the domain's current JupyterLab
# settings, replace only our entry in CustomImages, and send the rest back unchanged.
CURRENT="$(aws_capture "{}" sagemaker describe-domain --domain-id "$SM_DOMAIN_ID" \
  --query DefaultUserSettings.JupyterLabAppSettings --output json)"
USER_SETTINGS_JSON="$TMP_DIR/default-user-settings.json"
printf '%s' "$CURRENT" | jq --arg name "$SM_IMAGE_NAME" --argjson ver "$VERSION" --arg cfg "$SM_APP_IMAGE_CONFIG" '
  {JupyterLabAppSettings: ((. // {})
    | .CustomImages = ([(.CustomImages // [])[] | select(.ImageName != $name)]
        + [{ImageName: $name, ImageVersionNumber: $ver, AppImageConfigName: $cfg}]))}' > "$USER_SETTINGS_JSON"
show_file "$USER_SETTINGS_JSON"
note "Note: AWS's detach instructions say every app in the domain must be deleted before the"
note "custom-image list can be updated. If update-domain fails because apps are running, stop them"
note "(Studio > Running instances) and rerun this script."
aws_run sagemaker update-domain --domain-id "$SM_DOMAIN_ID" --default-user-settings "file://$USER_SETTINGS_JSON"
WAIT_OK="InService" WAIT_FAIL="Failed Update_Failed Delete_Failed Deleting" \
  wait_for "Domain $SM_DOMAIN_ID" 900 15 \
  aws sagemaker describe-domain --domain-id "$SM_DOMAIN_ID" --query Status --output text --region "$AWS_REGION"

if [[ -n "$SM_SPACE_NAME" ]]; then
  note ""
  note "== 5. Private JupyterLab space $SM_SPACE_NAME for $SM_USER_PROFILE on $SM_INSTANCE_TYPE"
  RESOURCE_SPEC_JSON="$TMP_DIR/resource-spec.json"
  jq -n --arg img "$IMAGE_ARN" --arg ver "$VERSION_ARN" --arg type "$SM_INSTANCE_TYPE" \
    '{SageMakerImageArn: $img, SageMakerImageVersionArn: $ver, InstanceType: $type}' > "$RESOURCE_SPEC_JSON"
  SPACE_SETTINGS_JSON="$TMP_DIR/space-settings.json"
  jq --slurpfile spec "$RESOURCE_SPEC_JSON" -n \
    '{AppType: "JupyterLab", JupyterLabAppSettings: {DefaultResourceSpec: $spec[0]}}' > "$SPACE_SETTINGS_JSON"
  if aws_probe sagemaker describe-space --domain-id "$SM_DOMAIN_ID" --space-name "$SM_SPACE_NAME"; then
    note "Space $SM_SPACE_NAME exists; its settings are not changed."
  else
    show_file "$SPACE_SETTINGS_JSON"
    aws_run sagemaker create-space --domain-id "$SM_DOMAIN_ID" --space-name "$SM_SPACE_NAME" \
      --ownership-settings "OwnerUserProfileName=$SM_USER_PROFILE" \
      --space-sharing-settings SharingType=Private \
      --space-settings "file://$SPACE_SETTINGS_JSON" \
      --tags "$TAGS"
    WAIT_OK="InService" WAIT_FAIL="Failed Update_Failed Delete_Failed Deleting" \
      wait_for "Space $SM_SPACE_NAME" 600 10 \
      aws sagemaker describe-space --domain-id "$SM_DOMAIN_ID" --space-name "$SM_SPACE_NAME" \
      --query Status --output text --region "$AWS_REGION"
  fi
  # A JupyterLab space has one app, and SageMaker requires it to be named "default".
  APP_STATUS="$(aws_capture_quiet "None" sagemaker describe-app --domain-id "$SM_DOMAIN_ID" \
    --space-name "$SM_SPACE_NAME" --app-type JupyterLab --app-name default \
    --query Status --output text || true)"
  if [[ "$APP_STATUS" == InService || "$APP_STATUS" == Pending ]]; then
    note "The space's JupyterLab app is already $APP_STATUS."
  else
    show_file "$RESOURCE_SPEC_JSON"
    aws_run sagemaker create-app --domain-id "$SM_DOMAIN_ID" --space-name "$SM_SPACE_NAME" \
      --app-type JupyterLab --app-name default \
      --resource-spec "file://$RESOURCE_SPEC_JSON" \
      --tags "$TAGS"
  fi
  note "The instance is billed from now until the app is stopped or deleted."
  WAIT_OK="InService" WAIT_FAIL="Failed Deleted Deleting" \
    WAIT_HINT="See FailureReason in: aws sagemaker describe-app --domain-id $SM_DOMAIN_ID --space-name $SM_SPACE_NAME --app-type JupyterLab --app-name default" \
    wait_for "JupyterLab app in $SM_SPACE_NAME" 1200 20 \
    aws sagemaker describe-app --domain-id "$SM_DOMAIN_ID" --space-name "$SM_SPACE_NAME" \
    --app-type JupyterLab --app-name default --query Status --output text --region "$AWS_REGION"
fi

note ""
note "Done."
note "Open it: SageMaker AI console > Studio > JupyterLab > your space (choose image '$SM_IMAGE_NAME'"
note "if you created the space yourself) > Run space > Open JupyterLab. Or get a sign-in URL:"
note "  aws sagemaker create-presigned-domain-url --domain-id $SM_DOMAIN_ID \\"
note "    --user-profile-name <profile> --space-name <space> --region $AWS_REGION"
note "Then, in a JupyterLab terminal in the space:"
note "  git clone https://github.com/project-delphi/marketing-statistics-workshop.git $SM_REPO_DIR"
note "Stop the space when you are not using it; teardown.sh removes everything this script created."

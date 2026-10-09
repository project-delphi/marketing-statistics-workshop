#!/usr/bin/env bash
# sagemaker-lifecycle-config.sh: the route without a custom image. Uploads lcc-on-start.sh as a
# Studio lifecycle configuration for JupyterLab and attaches it to the domain, so a private
# JupyterLab space on the default SageMaker Distribution image can run it at start.
#
# Documented, not run: the workshop authors have not run this script against AWS.
#
# Python labs only. The default image's Python version is set by its SageMaker Distribution
# version, and our pins are tested on Python 3.13 only; R is not supported on this route.
# Prefer the custom image (sagemaker-register-image.sh) when you can.
#
# Usage:
#   AWS_REGION=eu-west-1 SM_DOMAIN_ID=d-xxxxxxxxxxxx \
#   environment/aws/sagemaker-lifecycle-config.sh [--dry-run] [--help]
#
# Environment variables:
#   AWS_REGION         required.
#   SM_DOMAIN_ID       required. The Studio domain.
#   SM_LCC_NAME        lifecycle configuration name (default: mktstats-jupyterlab-on-start)
#   MKTSTATS_REF       branch or tag of the workshop repository the script clones (default: main)
#   SM_LCC_DEFAULT     0 (default): users pick the lifecycle configuration in their space settings;
#                      1: also make it the default for new private JupyterLab spaces
#                      (JupyterLabAppSettings.DefaultResourceSpec.LifecycleConfigArn).
#   PROJECT_TAG_VALUE  value of the Project tag (default: mktstats-workshop)
#
# Needs: aws (CLI v2), jq and openssl (also for --dry-run). An existing lifecycle configuration
# with the same name is reused unchanged; delete it with teardown.sh --lcc to change it.
#
# Docs checked on 2026-10-09:
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lifecycle-configurations.html
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lifecycle-configurations-create.html
#     (16,384-character limit, base64 with `openssl base64 -A`, app type JupyterLab,
#     JupyterLabAppSettings.LifecycleConfigArns)
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lifecycle-configurations-debug-timeout.html
#   https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-distribution.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-studio-lifecycle-config.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-studio-lifecycle-config.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/describe-domain.html
#   https://docs.aws.amazon.com/cli/latest/reference/sagemaker/update-domain.html
#     (LifecycleConfigArns: "To remove a lifecycle config, you must set LifecycleConfigArns to an
#     empty list"; DefaultResourceSpec.LifecycleConfigArn)

set -euo pipefail
# shellcheck source=_lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

for arg in "$@"; do
  common_flag "$arg" || die "Unknown argument: $arg (see --help)"
done
require_region
banner

require_var SM_DOMAIN_ID
need_cmd aws
command -v jq > /dev/null 2>&1 || die "jq is required (also for --dry-run)."
command -v openssl > /dev/null 2>&1 || die "openssl is required (also for --dry-run)."

SM_LCC_NAME="${SM_LCC_NAME:-mktstats-jupyterlab-on-start}"
MKTSTATS_REF="${MKTSTATS_REF:-main}"
SM_LCC_DEFAULT="${SM_LCC_DEFAULT:-0}"
[[ "$MKTSTATS_REF" =~ ^[A-Za-z0-9._/-]+$ ]] || die "MKTSTATS_REF must be a branch or tag name: $MKTSTATS_REF"

make_tmpdir
SCRIPT="$TMP_DIR/lcc-on-start.sh"
sed "s|^MKTSTATS_REF=.*|MKTSTATS_REF=\"$MKTSTATS_REF\"|" "$AWS_DIR/lcc-on-start.sh" > "$SCRIPT"
CHARS="$(wc -c < "$SCRIPT" | tr -d ' ')"
((CHARS <= 16384)) || die "lcc-on-start.sh is $CHARS characters; AWS allows 16,384."
CONTENT="$(openssl base64 -A -in "$SCRIPT")"
note "Lifecycle script: $CHARS characters (limit 16,384), MKTSTATS_REF=$MKTSTATS_REF"

note ""
note "== 1. Lifecycle configuration $SM_LCC_NAME"
if aws_probe sagemaker describe-studio-lifecycle-config --studio-lifecycle-config-name "$SM_LCC_NAME"; then
  note "$SM_LCC_NAME exists; reusing it unchanged."
else
  ARGS=(sagemaker create-studio-lifecycle-config --studio-lifecycle-config-name "$SM_LCC_NAME"
    --studio-lifecycle-config-content "$CONTENT"
    --studio-lifecycle-config-app-type JupyterLab
    --tags "Key=$PROJECT_TAG_KEY,Value=$PROJECT_TAG_VALUE")
  if [[ "$DRY_RUN" == 1 ]]; then
    # Show the command with the base64 shortened; the script it encodes is printed below.
    note "+ aws sagemaker create-studio-lifecycle-config --studio-lifecycle-config-name $SM_LCC_NAME --studio-lifecycle-config-content <base64 of the script below, ${#CONTENT} characters> --studio-lifecycle-config-app-type JupyterLab --tags Key=$PROJECT_TAG_KEY,Value=$PROJECT_TAG_VALUE --region $AWS_REGION"
    show_file "$SCRIPT"
  else
    aws_run "${ARGS[@]}"
  fi
fi
LCC_ARN="$(aws_capture "arn:aws:sagemaker:$AWS_REGION:<account-id>:studio-lifecycle-config/$SM_LCC_NAME" \
  sagemaker describe-studio-lifecycle-config --studio-lifecycle-config-name "$SM_LCC_NAME" \
  --query StudioLifecycleConfigArn --output text)"

note ""
note "== 2. Attach it to domain $SM_DOMAIN_ID (private JupyterLab spaces)"
# Send back the domain's current JupyterLab settings with only the lifecycle settings changed.
CURRENT="$(aws_capture "{}" sagemaker describe-domain --domain-id "$SM_DOMAIN_ID" \
  --query DefaultUserSettings.JupyterLabAppSettings --output json)"
USER_SETTINGS_JSON="$TMP_DIR/default-user-settings.json"
printf '%s' "$CURRENT" | jq --arg arn "$LCC_ARN" --arg makedefault "$SM_LCC_DEFAULT" '
  {JupyterLabAppSettings: ((. // {})
    | .LifecycleConfigArns = (((.LifecycleConfigArns // []) - [$arn]) + [$arn])
    | if $makedefault == "1"
      then .DefaultResourceSpec = ((.DefaultResourceSpec // {}) + {LifecycleConfigArn: $arn})
      else . end)}' > "$USER_SETTINGS_JSON"
show_file "$USER_SETTINGS_JSON"
aws_run sagemaker update-domain --domain-id "$SM_DOMAIN_ID" --default-user-settings "file://$USER_SETTINGS_JSON"
WAIT_OK="InService" WAIT_FAIL="Failed Update_Failed Delete_Failed Deleting" \
  wait_for "Domain $SM_DOMAIN_ID" 900 15 \
  aws sagemaker describe-domain --domain-id "$SM_DOMAIN_ID" --query Status --output text --region "$AWS_REGION"

note ""
note "Done. In Studio, create (or stop and edit) a private JupyterLab space on the default"
note "SageMaker Distribution image, choose the lifecycle configuration '$SM_LCC_NAME'"
if [[ "$SM_LCC_DEFAULT" == 1 ]]; then note "(preselected for new spaces),"; fi
note "then run the space. Watch ~/.mktstats-lcc.log in a JupyterLab terminal for the install."

#!/usr/bin/env bash
# build-push-ecr.sh: put the workshop image (environment/Dockerfile) into a private Amazon ECR
# repository, built for linux/amd64, ready for SageMaker Studio (sagemaker-register-image.sh).
# Either builds the image (default) or copies an existing one (SOURCE_IMAGE), e.g. the CI image
# from GHCR.
#
# Documented, not run: the workshop authors have not run this script against AWS.
#
# Usage:
#   AWS_REGION=eu-west-1 environment/aws/build-push-ecr.sh [--dry-run] [--help]
#
# Environment variables:
#   AWS_REGION         required. Must be the Region of your SageMaker domain (AWS requires the
#                      ECR repository and the domain to be in the same Region).
#   ECR_REPO           repository name (default: mktstats-env)
#   IMAGE_TAG          tag to push (default: the CI tag, the first 12 characters of the sha256 of
#                      environment/Dockerfile + requirements.txt + r-packages.txt, as image.yml does)
#   SOURCE_IMAGE       optional. Copy this image instead of building, for example
#                      ghcr.io/project-delphi/marketing-statistics-workshop/env:<tag>
#                      (the GHCR package may be private until the repository owner makes it public)
#   PROJECT_TAG_VALUE  value of the Project tag on the repository (default: mktstats-workshop)
#
# Needs: aws (CLI v2), docker with buildx. The repository is created with immutable tags, scan on
# push and AES256 encryption. A tag that is already in the repository is not pushed again.
#
# Docs checked on 2026-10-09 (every command and flag below was compared with these pages):
#   https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-byoi-how-to-prepare-image.html
#   https://docs.aws.amazon.com/cli/latest/reference/sts/get-caller-identity.html
#   https://docs.aws.amazon.com/cli/latest/reference/ecr/describe-repositories.html
#   https://docs.aws.amazon.com/cli/latest/reference/ecr/create-repository.html
#   https://docs.aws.amazon.com/cli/latest/reference/ecr/describe-images.html
#   https://docs.aws.amazon.com/cli/latest/reference/ecr/get-login-password.html
#   https://docs.docker.com/reference/cli/docker/buildx/build/ (--platform, --provenance, --sbom, --push)
#   https://docs.docker.com/reference/cli/docker/image/pull/ (--platform), .../image/tag/, .../image/push/
#   https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility

set -euo pipefail
# shellcheck source=_lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

for arg in "$@"; do
  common_flag "$arg" || die "Unknown argument: $arg (see --help)"
done
require_region
banner

ECR_REPO="${ECR_REPO:-mktstats-env}"
SOURCE_IMAGE="${SOURCE_IMAGE:-}"
[[ "$ECR_REPO" =~ ^[a-z0-9]+([._-][a-z0-9]+)*(/[a-z0-9]+([._-][a-z0-9]+)*)*$ ]] ||
  die "ECR_REPO must be lowercase letters, digits, '.', '_', '-' or '/': $ECR_REPO"

need_cmd aws
need_cmd docker

# The CI tag: image.yml runs `cat Dockerfile requirements.txt r-packages.txt | sha256sum | cut -c1-12`.
env_hash() {
  local files=("$REPO_ROOT/environment/Dockerfile" "$REPO_ROOT/environment/requirements.txt"
    "$REPO_ROOT/environment/r-packages.txt")
  if command -v sha256sum > /dev/null 2>&1; then
    cat "${files[@]}" | sha256sum | cut -c1-12
  else
    cat "${files[@]}" | shasum -a 256 | cut -c1-12
  fi
}
IMAGE_TAG="${IMAGE_TAG:-$(env_hash)}"

ACCOUNT_ID="$(aws_capture "<account-id>" sts get-caller-identity --query Account --output text)"
REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
IMAGE_URI="$REGISTRY/$ECR_REPO:$IMAGE_TAG"
note "Target image: $IMAGE_URI"

note ""
note "== 1. ECR repository"
if aws_probe ecr describe-repositories --repository-names "$ECR_REPO"; then
  note "Repository $ECR_REPO exists."
else
  aws_run ecr create-repository --repository-name "$ECR_REPO" \
    --image-tag-mutability IMMUTABLE \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 \
    --tags "Key=$PROJECT_TAG_KEY,Value=$PROJECT_TAG_VALUE"
fi

note ""
note "== 2. Push the image (skipped if the tag is already in the repository)"
if aws_probe ecr describe-images --repository-name "$ECR_REPO" --image-ids "imageTag=$IMAGE_TAG"; then
  note "Tag $IMAGE_TAG is already in $ECR_REPO; nothing to push."
else
  # docker login reads the password from the aws command through a pipe.
  note "+ aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REGISTRY"
  if [[ "$DRY_RUN" != 1 ]]; then
    aws ecr get-login-password --region "$AWS_REGION" |
      docker login --username AWS --password-stdin "$REGISTRY"
  fi
  if [[ -n "$SOURCE_IMAGE" ]]; then
    # Copy an existing linux/amd64 image. Untested on SageMaker: if create-image-version later
    # rejects the manifest, build with this script instead (the build path below pushes one
    # single-platform manifest with attestations turned off).
    run docker pull --platform linux/amd64 "$SOURCE_IMAGE"
    run docker tag "$SOURCE_IMAGE" "$IMAGE_URI"
    run docker push "$IMAGE_URI"
  else
    # On Apple Silicon this builds amd64 under emulation; that time has not been measured.
    # --provenance=false --sbom=false: push a plain image manifest, not an image index with
    # attestations (we have not verified whether SageMaker accepts an index).
    run docker buildx build --platform linux/amd64 \
      --file "$REPO_ROOT/environment/Dockerfile" \
      --tag "$IMAGE_URI" \
      --provenance=false --sbom=false \
      --push "$REPO_ROOT"
  fi
fi

note ""
note "Done. Next: register the image with your SageMaker domain:"
note "  AWS_REGION=$AWS_REGION IMAGE_URI=$IMAGE_URI SM_DOMAIN_ID=d-xxxxxxxxxxxx \\"
note "  SM_EXECUTION_ROLE_ARN=arn:aws:iam::$ACCOUNT_ID:role/<role> environment/aws/sagemaker-register-image.sh --dry-run"

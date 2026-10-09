These instructions and scripts have not been run on AWS by the workshop authors.

# Running the workshop on AWS (documented, not run)

Two routes use the workshop image (`environment/Dockerfile`: R 4.6.1, Python 3.13, JupyterLab,
Quarto): **SageMaker Studio** with the image as a custom image, and **EC2** running the image
behind an SSH tunnel. A third route, a **lifecycle configuration** on Studio's default image,
covers the Python labs only. Every command and JSON field in these files was checked against
the AWS documentation on 2026-10-09 (each script lists the pages it was checked against), but
none of it has been executed against an AWS account. `_variables.yml` lists these routes as the
readiness environments `aws-sagemaker` and `aws-ec2`, both documented only.

> [!WARNING]
> **Cost.** AWS bills for these resources while they exist, whether or not you are using them:
> EC2 instances and their EBS volumes, SageMaker Studio apps (a running space is a running
> instance) and space storage, ECR image storage, S3 storage and requests, and data transfer.
> Stop what you are not using and run [`teardown.sh`](#teardown) when you finish. Check the
> current prices for your Region on the [SageMaker AI pricing](https://aws.amazon.com/sagemaker/ai/pricing/)
> and [EC2 On-Demand pricing](https://aws.amazon.com/ec2/pricing/on-demand/) pages; this
> workshop does not quote prices.

## What is here

| File | What it does |
|---|---|
| `build-push-ecr.sh` | Builds the image for linux/amd64 (or copies the CI image from GHCR) and pushes it to a private ECR repository. |
| `sagemaker-register-image.sh` | Registers that image as a Studio JupyterLab custom image, attaches it to your domain, optionally starts a private space on it. |
| `sagemaker-lifecycle-config.sh`, `lcc-on-start.sh` | The route without a custom image: a lifecycle configuration that installs the pinned Python packages into Studio's default image. |
| `ec2-launch.sh`, `ec2-user-data.sh` | One Ubuntu 24.04 instance running JupyterLab from the image, reachable only through SSH or SSM port forwarding. |
| `teardown.sh` | Deletes what the scripts created, one flag per kind of resource, then lists what is left. |
| `iam/*.json` | Least-privilege IAM policies for each job, with placeholders. |
| `_lib.sh` | Helpers the scripts share (dry run, confirmation, Region check). |

Every script:

- refuses to run without `AWS_REGION` and passes `--region "$AWS_REGION"` to every `aws` call;
- takes its settings from environment variables, listed by `--help`;
- prints every command before running it; `--dry-run` prints them and runs nothing (no
  credentials needed, lookups print `<placeholders>`);
- asks before anything that deletes or starts a billed resource (`--yes` answers for you);
- tags what it creates `Project=mktstats-workshop`, which the IAM policies and `teardown.sh` rely on.

Start every route with a dry run, for example
`AWS_REGION=eu-west-1 EC2_SSH_CIDR=203.0.113.7/32 environment/aws/ec2-launch.sh --dry-run`.

**You need:** the [AWS CLI version 2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html),
`jq`, `openssl`, Docker with `buildx` (for `build-push-ecr.sh`), and for SSM the
[Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html).
The scripts are bash and were checked with shellcheck; they run under the bash 3.2 that macOS ships.

### Which route?

| Route | Python labs | R labs | You manage | Notes |
|---|---|---|---|---|
| (a) Studio + custom image | yes | yes | ECR image, Studio settings | Same image as CI and local Docker. Recommended. |
| (b) Studio + lifecycle configuration | yes | **no** | a startup script | Changes package versions in Studio's default image. |
| (c) EC2 + image | yes | yes | a Linux machine | Simplest to reason about; you patch and stop it yourself. |

**The image on GHCR.** CI publishes the image as
`ghcr.io/project-delphi/marketing-statistics-workshop/env:<hash>` and `:latest`, where `<hash>`
is the first 12 characters of the sha256 of `environment/Dockerfile`, `requirements.txt` and
`r-packages.txt` (`.github/workflows/image.yml`). GitHub makes a new package private by default;
until the repository owner makes it public, pulling it needs a GitHub token with `read:packages`
([GitHub: package visibility](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility)).
Building from `environment/Dockerfile` avoids GHCR altogether. Prefer a fixed `:<hash>` tag
to `:latest`, so everyone in a class runs the same environment.

## (a) SageMaker Studio with the workshop image

This covers the **new** SageMaker Studio only. Studio Classic "is no longer available for
onboarding" ([AWS: Studio Classic](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html);
[the new Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated.html)). You need a
Studio domain, a user profile in it and the domain's execution role; creating a domain is not
covered here ([AWS: domain quick setup](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html)).

**Why the image fits Studio.** AWS's requirements for a JupyterLab custom image
([specifications](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-byoi-specs.html),
[JupyterLab custom images](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-admin-guide-custom-images.html))
and how the workshop image meets them:

| Requirement | Workshop image |
|---|---|
| User ID 1000, group ID 100 | Runs as `sagemaker-user`, 1000:100. |
| The space's EBS volume is mounted on `/home/sagemaker-user` (whatever the image had there is hidden) | Python lives in `/opt/venv`, the R kernel is registered system-wide, so nothing the labs need is in `$HOME`. |
| JupyterLab on port 8888 with base URL `jupyterlab/default`; health check `jupyterlab/default/api/status` | Set by the app image config (below). |
| Token and password authentication off, all origins allowed (Studio does the sign-in) | Set by the app image config. |
| linux/amd64 | `build-push-ecr.sh` builds for linux/amd64. |

The app image config starts the container with entrypoint `jupyter-lab` (AWS allows one
entrypoint item) and these arguments, taken from AWS's sample Dockerfile:

```text
--ip=0.0.0.0 --port=8888 --no-browser --ServerApp.base_url=/jupyterlab/default
--ServerApp.token= --ServerApp.allow_origin=* --ServerApp.root_dir=/home/sagemaker-user
```

We ran exactly this command locally in a build of the workshop image (linux/arm64, Docker
Desktop, 2026-10-09): `jupyterlab/default/api/status` answered 200 without a token and both
kernels (`python3`, `ir`) were listed. Jupyter Server 2.20 prints a deprecation warning for
`ServerApp.token`; it still works. That is a local check, not a SageMaker run.

### 1. Push the image to ECR

```bash
AWS_REGION=eu-west-1 environment/aws/build-push-ecr.sh --dry-run
AWS_REGION=eu-west-1 environment/aws/build-push-ecr.sh
# or copy the CI image instead of building:
AWS_REGION=eu-west-1 SOURCE_IMAGE=ghcr.io/project-delphi/marketing-statistics-workshop/env:<hash> \
  environment/aws/build-push-ecr.sh
```

The ECR repository must be in the domain's Region
([AWS: push to ECR](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-byoi-how-to-prepare-image.html)).
The script creates it with immutable tags, scan on push and AES256 encryption, and tags the
image with the CI hash, so a tag always means one environment. The build pushes a single
linux/amd64 manifest (`--provenance=false --sbom=false`); we have not checked whether SageMaker
accepts the image index (manifest list) that the GHCR copy may carry, so if `create-image-version`
rejects a copied image, build instead. A cold build took about 16 minutes on an Apple Silicon
laptop for linux/arm64 (DECISIONS.md, S4); an amd64 build there runs under emulation and has
not been timed.

### 2. Register it with your domain

```bash
AWS_REGION=eu-west-1 SM_DOMAIN_ID=d-xxxxxxxxxxxx \
SM_EXECUTION_ROLE_ARN=arn:aws:iam::111122223333:role/<execution-role> \
IMAGE_URI=111122223333.dkr.ecr.eu-west-1.amazonaws.com/mktstats-env:<hash> \
  environment/aws/sagemaker-register-image.sh --dry-run
```

It runs `create-image`, `create-image-version` (waits for `CREATED`), `create-app-image-config`
(`JupyterLabAppImageConfig` with `FileSystemConfig` 1000/100 on `/home/sagemaker-user` and the
`ContainerConfig` above) and `update-domain`. `update-domain` replaces the domain's list of
JupyterLab custom images, so the script reads the current settings with `describe-domain`,
swaps in only its own entry under `DefaultUserSettings.JupyterLabAppSettings.CustomImages`, and
sends the rest back unchanged ([AWS: container configuration](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-byoi-how-to-container-configuration.html)).
Two things to know:

- AWS's detach instructions say every app in the domain must be deleted before the custom-image
  list can be updated. If `update-domain` fails because apps are running, stop them and rerun.
- `DefaultUserSettings` apply to **private** spaces. Shared spaces take their images from
  `DefaultSpaceSettings`, which the script does not change.

The same script can start a private JupyterLab space on the image: set `SM_USER_PROFILE`,
`SM_SPACE_NAME` and optionally `SM_INSTANCE_TYPE` (default `ml.m5.xlarge`). The app in a
JupyterLab space must be named `default`.

### 3. Open a space on the image

In the SageMaker AI console, open Studio, choose **JupyterLab**, create a space (or stop an
existing one), pick the image `mktstats-env` under **Image**, and choose **Run space**, then
**Open JupyterLab** ([AWS: launch a custom image](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-byoi-how-to-launch.html)).
From a terminal you can also get a sign-in link:
`aws sagemaker create-presigned-domain-url --domain-id <id> --user-profile-name <profile> --space-name <space> --region <region>`.

### 4. Clone the workshop and use its code

In a JupyterLab terminal in the space:

```bash
git clone https://github.com/project-delphi/marketing-statistics-workshop.git \
  /home/sagemaker-user/marketing-statistics-workshop
```

The clone lives on the space's volume and survives restarts. The app image config already sets,
for the server and every kernel:

```text
MKTSTATS_REPO_ROOT=/home/sagemaker-user/marketing-statistics-workshop
PYTHONPATH=/home/sagemaker-user/marketing-statistics-workshop/src
MKTSTATS_CACHE=/home/sagemaker-user/.cache/mktstats
```

so the notebooks use `mktstats` and the R helpers from the clone, not a copy from GitHub, as
in the local Docker setup (`setup.qmd`) and CI. To clone somewhere else, rerun the register
script with `SM_REPO_DIR=<path>` and a new `SM_APP_IMAGE_CONFIG` name (an existing config is
not changed), or set the variables at the top of a notebook before its first cell:
Python `import os, sys; os.environ["MKTSTATS_REPO_ROOT"] = "<path>"; sys.path.insert(0, "<path>/src")`,
R `Sys.setenv(MKTSTATS_REPO_ROOT = "<path>")`.

### Instance type

CI runs the notebooks on GitHub-hosted runners (4 CPUs and 16 GB of RAM for public
repositories, per [GitHub's runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
read 2026-10-09); a 4 vCPU / 16 GB CPU instance is the closest equivalent; no AWS timing has been
measured. In Studio that is `ml.m5.xlarge` (4 vCPU, 16 GiB, per
[AWS's instance table](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html)),
the scripts' default. The labs are sized for Colab's free runtime (2 CPUs, 13.6 GB, DECISIONS.md
S1); `ml.t3.medium` (2 vCPU, 4 GiB) has much less memory than that. No GPU is needed: the models
run on the CPU.

**Stop the space when you are not using it.** AWS documents idle shutdown for the SageMaker
Distribution image (v2.0 or newer) ([AWS: idle shutdown](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-idle-shutdown.html));
the workshop image is not that image, so do not count on it.

## (b) Lifecycle configuration instead of a custom image

A lifecycle configuration is a shell script Studio runs each time a space starts
([AWS: lifecycle configurations](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lifecycle-configurations.html)).
`sagemaker-lifecycle-config.sh` uploads `lcc-on-start.sh` (base64, under AWS's 16,384-character
limit), adds it to the domain's JupyterLab settings, and with `SM_LCC_DEFAULT=1` preselects it for
new private spaces:

```bash
AWS_REGION=eu-west-1 SM_DOMAIN_ID=d-xxxxxxxxxxxx environment/aws/sagemaker-lifecycle-config.sh --dry-run
```

**What it does.** At space start it clones the repository into `~/marketing-statistics-workshop`
(an existing clone is left alone) and installs the lab packages (`pymc-marketing`, `econml`,
`nutpie` at the versions in `_variables.yml`, with `environment/requirements.txt` as constraints,
the same way the Colab install cell does) into the default image's Python. AWS stops a
lifecycle script after 5 minutes
([AWS: timeout](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lifecycle-configurations-debug-timeout.html)),
so the install runs in the background with `nohup`; follow it in `~/.mktstats-lcc.log`. The
script's own output goes to CloudWatch, log group `/aws/sagemaker/studio`, stream
`<domain-id>/<space-name>/JupyterLab/default/LifecycleConfigOnStart`.

**What it cannot do.**

- **R is not supported on this route.** The script installs no R, no R packages and no R kernel;
  use the custom image or EC2 for the R labs.
- It changes package versions inside Studio's default image: the constraints pin numpy, pandas,
  scikit-learn and others to the Colab-matched versions, which may break other tools in that
  environment.
- The default image's Python version depends on the SageMaker Distribution version
  ([AWS: SageMaker Distribution](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-distribution.html));
  the pins are tested on Python 3.13 only. If the install fails to resolve, the log says so; use
  the custom image.
- It does not set `MKTSTATS_REPO_ROOT`. To make the labs use the clone, set it at the top of a
  notebook as in step 4 of route (a).

## (c) EC2 with the workshop image

`ec2-launch.sh` launches one Ubuntu Server 24.04 (amd64) instance and passes it
`ec2-user-data.sh`, which at first boot installs Docker Engine from Docker's apt repository,
clones the repository to `/opt/mktstats/workshop`, writes a random JupyterLab token, and runs the
image as the systemd service `mktstats-jupyter`. The container's port 8888 is published on the
instance's **127.0.0.1 only** (`-p 127.0.0.1:8888:8888`): nothing listens on a public interface.
You reach JupyterLab through one of:

- **SSH tunnel** (`EC2_ACCESS=ssh`, default). The security group allows SSH (port 22) from
  `EC2_SSH_CIDR` only, normally your public address as a `/32`; the script refuses `0.0.0.0/0`.
  It creates an ed25519 key pair and saves the private key to `~/.ssh/mktstats-workshop.pem`.
- **SSM Session Manager port forwarding** (`EC2_ACCESS=ssm`). No inbound rules at all. The
  instance needs an instance profile whose role has the AWS managed policy
  `AmazonSSMManagedInstanceCore` ([AWS: instance permissions](https://docs.aws.amazon.com/systems-manager/latest/userguide/setup-instance-permissions.html));
  create it once and pass its name as `EC2_INSTANCE_PROFILE`. SSM Agent is preinstalled on
  Ubuntu Server 24.04 AMIs ([AWS](https://docs.aws.amazon.com/systems-manager/latest/userguide/ami-preinstalled-agent.html)).
  The instance needs outbound HTTPS to the Systems Manager endpoints (a public IP or a NAT, or VPC
  endpoints) ([AWS: prerequisites](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-prerequisites.html)).

```bash
AWS_REGION=eu-west-1 EC2_SSH_CIDR=203.0.113.7/32 environment/aws/ec2-launch.sh --dry-run
AWS_REGION=eu-west-1 EC2_SSH_CIDR=203.0.113.7/32 \
  EC2_IMAGE=ghcr.io/project-delphi/marketing-statistics-workshop/env:<hash> environment/aws/ec2-launch.sh
```

The AMI comes from Canonical's public parameter
`/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id`
([Ubuntu: find images](https://documentation.ubuntu.com/aws/en/latest/aws-how-to/instances/find-ubuntu-images/)).
The root volume is gp3, encrypted, 40 GiB by default (`EC2_VOLUME_GB`) and deleted with the
instance. The instance requires IMDSv2 with a hop limit of 1, so processes inside the Docker
container cannot get a metadata token and therefore cannot read the instance role's credentials
(AWS suggests a hop limit of 2 only when containers need instance metadata;
[AWS: IMDS options](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-IMDS-new-instances.html)).

**Connect** (the user on Ubuntu AMIs is `ubuntu`). The first boot pulls a multi-GB image; how
long that takes on EC2 has not been measured.

```bash
# progress and token
ssh -i ~/.ssh/mktstats-workshop.pem ubuntu@<public-ip> 'journalctl -u mktstats-jupyter -n 20'
ssh -i ~/.ssh/mktstats-workshop.pem ubuntu@<public-ip> 'sudo cat /etc/mktstats/jupyter-token'
# tunnel: leave it open, then browse to http://127.0.0.1:8888/lab?token=<token>
ssh -i ~/.ssh/mktstats-workshop.pem -N -L 8888:127.0.0.1:8888 ubuntu@<public-ip>

# or with SSM
aws ssm start-session --target <instance-id> --region <region>      # then: sudo cat /etc/mktstats/jupyter-token
aws ssm start-session --target <instance-id> --region <region> \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8888"],"localPortNumber":["8888"]}'
```

The token is written once, at first boot, to `/etc/mktstats/jupyter-token` (readable by root
and by the container user) and passed to Jupyter Server through `JUPYTER_TOKEN_FILE`; JupyterLab's
own log shows `token=...`, not the token. We ran the service's `docker run` command locally
(linux/arm64, 2026-10-09): JupyterLab answered 403 without the token and 200 with it, on
127.0.0.1 only. The user-data script as a whole has not been run on EC2.

**Sizing.** CI runs the notebooks on GitHub-hosted runners (4 CPUs and 16 GB of RAM for public
repositories, per [GitHub's runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners));
a 4 vCPU / 16 GB CPU instance is the closest equivalent; no AWS timing has been measured. The
default is `m7i.xlarge` (4 vCPU, 16 GiB, [AWS](https://aws.amazon.com/ec2/instance-types/m7i/)).
The image is linux/amd64, so pick an x86_64 type, not Graviton. No GPU is needed.

**If the GHCR image is private,** the pull fails with `denied` or `unauthorized` in
`journalctl -u mktstats-jupyter`. Do not put a GitHub token in user data: AWS lets anyone allowed
to describe the instance view its user data, and every process on the instance can read it
through the metadata service ([AWS: user data](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html)).
Instead, log in on the instance with a token that has `read:packages`
([GitHub: container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)),
then restart the service:

```bash
export CR_PAT=<token>; echo "$CR_PAT" | sudo docker login ghcr.io -u <github-user> --password-stdin
sudo systemctl restart mktstats-jupyter
```

Or ask the repository owner to make the package public.

**Stop or terminate it.** A stopped instance still pays for its volume; `teardown.sh --ec2`
terminates it.

## IAM: least privilege

Each policy in `iam/` covers one job. Replace the placeholders before use:

| Placeholder | Meaning |
|---|---|
| `ACCOUNT_ID`, `REGION` | your account and Region |
| `REPO_NAME` | ECR repository (`ECR_REPO`, default `mktstats-env`) |
| `DOMAIN_ID` | the Studio domain |
| `IMAGE_NAME`, `APP_IMAGE_CONFIG_NAME`, `LCC_NAME` | `SM_IMAGE_NAME`, `SM_APP_IMAGE_CONFIG`, `SM_LCC_NAME` (defaults `mktstats-env`, `mktstats-env-jupyterlab`, `mktstats-jupyterlab-on-start`) |
| `ROLE_NAME` | the domain's SageMaker execution role |
| `INSTANCE_ROLE_NAME` | the EC2 instance role for SSM |
| `BUCKET`, `PREFIX` | the S3 data bucket and the key prefix the workshop may use |

The tag value `mktstats-workshop` is written into the EC2 and SSM policies; change it there if
you set `PROJECT_TAG_VALUE`. Actions, resource ARN formats and condition keys were checked
against the [Service Authorization Reference](https://docs.aws.amazon.com/service-authorization/latest/reference/reference_policies_actions-resources-contextkeys.html)
(SageMaker, ECR, EC2, SSM, S3, Resource Groups Tagging API, IAM pages) on 2026-10-09. No policy
has been tested with the IAM policy simulator or a real call. `sts get-caller-identity`, which
the scripts use to find the account ID, needs no permission.

**1. `image-builder-ecr.json`** (who runs `build-push-ecr.sh`; also `teardown.sh --ecr-repo`)

- `EcrLogin`: `ecr:GetAuthorizationToken` cannot be limited to a repository, so it is `*`.
- `EcrCreateAndInspectWorkshopRepository`: create the one repository (with its tag, which needs
  `ecr:TagResource`) and check whether it and a tag exist.
- `EcrPushToWorkshopRepository`: the six layer and manifest actions AWS lists for a push to a
  specific repository ([AWS: push permissions](https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push-iam.html)).
- `EcrDeleteWorkshopRepository`: only for teardown; remove it if someone else cleans up.

**2. `sagemaker-admin.json`** (who runs `sagemaker-register-image.sh`, `sagemaker-lifecycle-config.sh`
and the SageMaker parts of `teardown.sh`)

- `WorkshopImage`, `WorkshopImageVersions`: create, read and delete one image and its versions.
- `WorkshopAppImageConfig`, `WorkshopLifecycleConfig`: the same for one app image config and one
  lifecycle configuration.
- `OneDomain`: read and update one domain (attach and detach the image and the lifecycle
  configuration). `UpdateDomain` is a single action, so this also allows changing the domain's
  other settings; condition keys such as `sagemaker:ImageArns` could narrow it but are not used
  here, because the scripts send the domain's existing custom images back with their own.
- `SpacesAndAppsInThatDomain`: create, read and delete spaces and apps in that domain only.
- `SignInUrlForThatDomain`: `create-presigned-domain-url` for user profiles in that domain.
- `TagWhatTheScriptsCreate`: `sagemaker:AddTags` is checked when a create call carries tags,
  so it is allowed only on the resources above.
- `ListCallsWithoutResourceScope`: the `List*` calls (and `tag:GetResources` for the teardown
  check) do not support resource-level permissions, so they are `*`; they only read.
- `PassOnlyTheExecutionRoleToSageMaker`: `create-image` passes a role (`iam:PassRole`); this
  allows only the domain's execution role and only to `sagemaker.amazonaws.com`.

**3. `sagemaker-execution-role.json`** (attach to the execution role `ROLE_NAME`)

- `ListOnlyTheWorkshopPrefix`: `s3:ListBucket` on the bucket, limited by `s3:prefix` to
  `PREFIX/*`, which `aws s3 sync` needs to compare files.
- `ReadWriteObjectsUnderThePrefix`: get, put and delete objects under `PREFIX/` only.
- `EcrLoginForImagePull`, `PullTheWorkshopImage`: what AWS's example execution-role policy grants
  for pulling an image from specific repositories
  ([AWS: execution roles](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html)).
  The role passed to `create-image` must be able to pull the image.

This policy adds data access and image pull to a role that already exists with the domain; it
is not everything a domain's execution role needs (AWS's procedure for creating an execution
role attaches `AmazonSageMakerFullAccess`). We have not worked out the minimum set for running a
space.

**4. `ec2-launcher.json`** (who runs `ec2-launch.sh` and `teardown.sh --ec2`)

- `DescribeCallsWithoutResourceScope`: the `Describe*` calls (and `tag:GetResources`) do not
  support resource-level permissions; they only read.
- `ReadTheUbuntu2404AmiParameter`: `ssm:GetParameter` on Canonical's one public parameter
  (public parameters have no account ID in their ARN).
- `RunInstancesUntaggedResources`: `RunInstances` also authorizes the AMI, subnet, network
  interface, security group and key pair, which are not tagged at launch; this follows AWS's
  example ([AWS: example policies](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ExamplePolicies_EC2.html)).
- `RunInstancesOnlyWithTheProjectTag`: the instance and its volume can be created only if the
  request tags them `Project=mktstats-workshop`.
- `TagOnlyWhileCreating`: `ec2:CreateTags` only as part of `RunInstances`, `CreateSecurityGroup`
  or `CreateKeyPair` (`ec2:CreateAction`), so existing resources cannot be retagged into scope
  ([AWS: tag on create](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/supported-iam-actions-tagging.html)).
- `TerminateOnlyTaggedInstances`: terminate only instances tagged `Project=mktstats-workshop`.
- `CreateSecurityGroupOnlyWithTheProjectTag`, `CreateSecurityGroupInAnyVpc`: a security group
  can be created only with the tag; the VPC is a separate resource in that call.
- `ChangeAndDeleteOnlyTaggedSecurityGroups`: add the SSH rule to, and delete, only tagged groups.
- `TheRuleThatAnIngressAuthorizationCreates`: `AuthorizeSecurityGroupIngress` also names a
  `security-group-rule` resource; allowing it does not widen access, because the group itself
  must still be tagged.
- `CreateKeyPairOnlyWithTheProjectTag`, `DeleteOnlyTaggedKeyPairs`: the same for key pairs.

To also restrict instance types, add a statement with the `ec2:InstanceType` condition on
`instance/*` (see AWS's example policies).

**5. `ec2-ssm-session.json`** (optional, for `EC2_ACCESS=ssm`; add it to the launcher's permissions)

- `PassOnlyTheInstanceRoleToEc2`: launching with an instance profile passes its role to
  `ec2.amazonaws.com`; only `INSTANCE_ROLE_NAME` is allowed.
- `SessionsOnlyToTaggedInstances`: start sessions only on instances tagged
  `Project=mktstats-workshop` (`ssm:resourceTag`, as in
  [AWS's Session Manager examples](https://docs.aws.amazon.com/systems-manager/latest/userguide/getting-started-restrict-access-quickstart.html)).
- `OnlyPortForwardingAndTheDefaultShell`: the documents a session may use. The account part of
  the AWS-owned `AWS-StartPortForwardingSession` ARN is a wildcard because we could not confirm
  its exact ARN form in the docs.
- `OwnSessionsOnly`: end, resume and open the data channel of your own sessions only.

**6. `s3-data-bucket-admin.json`** (optional, for whoever creates and deletes the data bucket)

- `CreateConfigureListAndDeleteTheBucket`: create the bucket, block public access, set default
  encryption, list it (also needed by `head-bucket` and `rb --force`) and delete it.
- `UploadDownloadAndDeleteObjects`: fill it, read it, and empty it before deletion.

## S3 data bucket (optional)

Use this when data must stay inside your company's AWS account. **mktstats does not read S3.**
Its loaders download files and cache them in the directory named by `MKTSTATS_CACHE`; the R
helpers do this today, and the Python loaders (`src/mktstats/data.py`) are still to be written,
so check that they honor `MKTSTATS_CACHE` before relying on this. The approach: fill the cache
once where downloads are allowed, copy it to S3, and copy it back into `MKTSTATS_CACHE` wherever
the labs run. That mirrors whatever layout the loaders use. With `MKTSTATS_REPO_ROOT` set, the
synthetic datasets come from the clone's `data/` and need no cache.

Create the bucket (in `us-east-1`, leave out `--create-bucket-configuration`):

```bash
aws s3api create-bucket --bucket BUCKET --region REGION \
  --create-bucket-configuration LocationConstraint=REGION
aws s3api put-public-access-block --bucket BUCKET --region REGION \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket BUCKET --region REGION \
  --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

New buckets already block public access and encrypt new objects with SSE-S3
([AWS: Block Public Access](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-control-block-public-access.html),
[default encryption](https://docs.aws.amazon.com/AmazonS3/latest/userguide/default-bucket-encryption.html));
the commands make that explicit. Versioning stays off, which `teardown.sh --s3-bucket` relies on.

Copy the cache up, then down where the labs run:

```bash
aws s3 sync "$MKTSTATS_CACHE" s3://BUCKET/PREFIX/cache/ --region REGION
aws s3 sync s3://BUCKET/PREFIX/cache/ "$MKTSTATS_CACHE" --region REGION
```

- **In a Studio space with the workshop image:** the image has no AWS CLI. In a JupyterLab
  terminal, `uv tool install awscli` installs AWS CLI version 1 into `~/.local/bin` on the space's
  volume (we tried this locally in the image as `sagemaker-user`: it installed `aws-cli/1.46.1`);
  then run `~/.local/bin/aws s3 sync ...` with `MKTSTATS_CACHE=/home/sagemaker-user/.cache/mktstats`.
  Studio provides the execution role's credentials to the container
  (`AWS_CONTAINER_CREDENTIALS_RELATIVE_URI`, [AWS specs](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-byoi-specs.html));
  the role needs `iam/sagemaker-execution-role.json`. Not tested on SageMaker.
- **On EC2:** run the sync on the instance (not in the container), into `/opt/mktstats/cache`,
  which the container sees as `MKTSTATS_CACHE`. The instance needs an instance profile with the
  same S3 statements and the AWS CLI installed on the host.

## Teardown

`teardown.sh` deletes nothing unless you name it with a flag, asks before each deletion, skips
what is already gone, and always runs in this order:

1. `--ec2`: terminate every instance tagged `Project=mktstats-workshop` and wait (their root
   volumes go with them); then delete the tagged security group and key pair. Your local
   private key file is kept; delete it yourself.
2. `--sagemaker-apps`: delete the JupyterLab apps that run the workshop image and wait; then the
   space `SM_SPACE_NAME`, if set (deleting a space deletes the files stored in it).
3. `--lcc`: remove the lifecycle configuration from the domain, then delete it.
4. `--sagemaker-image`: remove the image from the domain's custom images (AWS requires this
   before the image is deleted, and every app in the domain must be deleted before the list can
   change), delete the app image config, the image versions and the image. The copy in ECR stays.
5. `--ecr-repo`: delete the ECR repository and every image in it (`--force`).
6. `--s3-bucket`: delete `S3_BUCKET` and every object in it (`aws s3 rb --force`; it cannot
   delete old object versions, which is why versioning stays off).

```bash
AWS_REGION=eu-west-1 environment/aws/teardown.sh --ec2 --dry-run
AWS_REGION=eu-west-1 SM_DOMAIN_ID=d-xxxxxxxxxxxx SM_SPACE_NAME=<space> \
  environment/aws/teardown.sh --sagemaker-apps --lcc --sagemaker-image
AWS_REGION=eu-west-1 S3_BUCKET=BUCKET environment/aws/teardown.sh --ecr-repo --s3-bucket
```

**Check that nothing is left.** The script ends with these read-only checks; empty output or a
"not found" error means nothing is left:

```bash
aws resourcegroupstaggingapi get-resources --tag-filters Key=Project,Values=mktstats-workshop \
  --query 'ResourceTagMappingList[].ResourceARN' --output text --region REGION
aws ec2 describe-instances --filters Name=tag:Project,Values=mktstats-workshop \
  Name=instance-state-name,Values=pending,running,shutting-down,stopping,stopped \
  --query 'Reservations[].Instances[].InstanceId' --output text --region REGION
aws ec2 describe-volumes --filters Name=tag:Project,Values=mktstats-workshop --query 'Volumes[].VolumeId' --output text --region REGION
aws ec2 describe-security-groups --filters Name=tag:Project,Values=mktstats-workshop --query 'SecurityGroups[].GroupId' --output text --region REGION
aws ec2 describe-key-pairs --filters Name=tag:Project,Values=mktstats-workshop --query 'KeyPairs[].KeyName' --output text --region REGION
aws sagemaker list-apps --domain-id-equals DOMAIN_ID --query "Apps[?Status!='Deleted'].[SpaceName,AppType,AppName,Status]" --output text --region REGION
aws sagemaker list-spaces --domain-id-equals DOMAIN_ID --query 'Spaces[].SpaceName' --output text --region REGION
aws sagemaker list-images --name-contains mktstats-env --query 'Images[].ImageName' --output text --region REGION
aws sagemaker list-app-image-configs --name-contains mktstats-env-jupyterlab --query 'AppImageConfigs[].AppImageConfigName' --output text --region REGION
aws sagemaker list-studio-lifecycle-configs --name-contains mktstats-jupyterlab-on-start --query 'StudioLifecycleConfigs[].StudioLifecycleConfigName' --output text --region REGION
aws ecr describe-repositories --repository-names mktstats-env --region REGION
aws s3api head-bucket --bucket BUCKET --region REGION
```

The tag search finds only what the scripts tagged; the S3 bucket from the commands above is not
tagged. The Studio domain, user profiles, execution role and instance profile are yours and are
never deleted. Check the Billing console's cost breakdown a day later as well.

## What has and has not been verified

- **Checked against documentation (2026-10-09):** every `aws` command, flag, JSON field and
  value in these scripts and this page (AWS CLI 2.37.11 reference pages, the SageMaker, EC2,
  Systems Manager and S3 guides, the Service Authorization Reference), Canonical's AMI parameter,
  Docker's Ubuntu install steps, GitHub's runner sizes and package visibility. Each script lists
  its pages.
- **Run locally, not on AWS (2026-10-09, linux/arm64 build of the workshop image):** the Studio
  container arguments (health check answers without a token; both kernels present); the EC2
  service's `docker run` line (token required, published on 127.0.0.1 only); `uv tool install
  awscli` as `sagemaker-user`; every script with `--dry-run`; shellcheck on every script.
- **Not verified:** anything that needs an AWS account: that SageMaker accepts the image and
  starts it, that `update-domain` merges settings as described, the IAM policies, the user data
  on a real instance, timings, and whether SageMaker accepts an image index copied from GHCR.

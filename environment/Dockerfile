# syntax=docker/dockerfile:1.7
# One image for local work, CI and AWS: R 4.6.1 + Python 3.13 + JupyterLab + Quarto.
# It matches Colab's runtime as of 2026-10 (Ubuntu 24.04, R 4.6.1, Python 3.13).
# Build:  docker build -f environment/Dockerfile -t mktstats-env .
# Run:    docker run --rm -p 8888:8888 -v "$PWD":/home/sagemaker-user/work mktstats-env
FROM rocker/r-ver:4.6.1

ARG TARGETARCH
ARG QUARTO_VERSION=1.10.19
ARG UV_VERSION=0.11.32
# Dated Posit Public Package Manager snapshot: binaries for amd64 and arm64 on noble,
# the same snapshot the Colab R notebooks install from.
ARG P3M_SNAPSHOT=2026-10-01

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    MPLBACKEND=Agg \
    RETICULATE_PYTHON=/opt/venv/bin/python

# System libraries: GSL for CLVTools, ZeroMQ for IRkernel, git for pip installs from GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git libgsl-dev libzmq3-dev libxml2-dev libcurl4-openssl-dev \
        libssl-dev libfontconfig1-dev libfreetype-dev libpng-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Quarto (bundles pandoc).
RUN curl -fsSL -o /tmp/quarto.deb \
        "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-${TARGETARCH}.deb" \
    && apt-get update && apt-get install -y --no-install-recommends /tmp/quarto.deb \
    && rm -rf /tmp/quarto.deb /var/lib/apt/lists/*

# Python 3.13 in /opt/venv, pinned to environment/requirements.txt.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /usr/local/bin/
COPY environment/requirements.txt /tmp/requirements.txt
RUN uv venv --python 3.13 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache -r /tmp/requirements.txt

# R packages from the dated snapshot (binaries; HTTPUserAgent is set by rocker's Rprofile.site).
RUN echo "options(repos = c(CRAN = 'https://p3m.dev/cran/__linux__/noble/${P3M_SNAPSHOT}'))" \
        >> /usr/local/lib/R/etc/Rprofile.site
COPY environment/r-packages.txt /tmp/r-packages.txt
# A dependency whose download fails is only a warning to install.packages, so the check
# covers the whole dependency tree, not just the packages named in r-packages.txt.
RUN Rscript -e 'options(timeout = 900); \
        pk <- setdiff(readLines("/tmp/r-packages.txt"), ""); pk <- pk[!startsWith(pk, "#")]; \
        install.packages(pk, Ncpus = parallel::detectCores()); \
        deps <- unlist(tools::package_dependencies(pk, recursive = TRUE, \
                       which = c("Depends", "Imports", "LinkingTo"))); \
        need <- setdiff(unique(c(pk, deps)), rownames(installed.packages())); \
        if (length(need)) install.packages(need); \
        missing <- setdiff(unique(c(pk, deps)), rownames(installed.packages())); \
        if (length(missing)) stop("Not installed: ", paste(missing, collapse = ", "))' \
    && Rscript -e 'IRkernel::installspec(user = FALSE, name = "ir", displayname = "R")'

# GitHub-only packages from archive tarballs at the commits pinned in _variables.yml
# (github_packages); no GitHub API call, the same way the Colab R notebooks install them.
ARG AUGSYNTH_SHA=7e70072232fe75057aa8b5d7af08c7e1aef53ae8
ARG GEOLIFT_SHA=db34ea4299ff0e28515ebac502b78c076c93c905
RUN Rscript -e 'options(timeout = 900); \
        install.packages(sprintf("https://github.com/ebenmichael/augsynth/archive/%s.tar.gz", Sys.getenv("AUGSYNTH_SHA")), repos = NULL, type = "source"); \
        install.packages(sprintf("https://github.com/facebookincubator/GeoLift/archive/%s.tar.gz", Sys.getenv("GEOLIFT_SHA")), repos = NULL, type = "source"); \
        stopifnot(requireNamespace("augsynth"), requireNamespace("GeoLift"))'

# SageMaker Studio custom images expect UID 1000 / GID 100 and port 8888 (environment/aws/README.md).
RUN useradd -m -u 1000 -g 100 -s /bin/bash sagemaker-user
USER 1000:100
WORKDIR /home/sagemaker-user
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

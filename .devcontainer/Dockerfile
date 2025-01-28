ARG VARIANT="ubuntu22.04" 
ARG CUDA_VERSION="12.6.2"
ARG UV_VERSION="0.5.7"
ARG USERNAME="user"
ARG USER_UID="1000"
ARG USER_GID=${USER_UID}
ARG PYTHON_VERSION='3.10'
ARG GO_VERSION='23.4'
ARG POSTGRES_VERSION='16'
ARG NODE_VERSION='v0.40.1'
ARG NVM_VERSION='22.0.0'
ARG PROJECT_NAME='project'
ARG PROJDIR

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_image
FROM golang:${GO_VERSION}-alpine AS go_image

# Start from a CUDA development image
FROM nvidia/cuda:${CUDA_VERSION}-devel-${VARIANT} AS build-env
# FROM ubuntu:${UBUNTU_VARIANT} AS build-env
COPY --from=uv_image /uv /uvx /bin/
COPY --from=go_image /usr/local/go/ /usr/local/go/
ENV PATH="/usr/local/go/bin:${PATH}"
 
ARG USERNAME
ARG USER_UID
ARG USER_GID
ARG PYTHON_VERSION
ARG GO_VERSION
ARG POSTGRES_VERSION
ARG NODE_VERSION
ARG NVM_VERSION
ARG PROJECT_NAME

ARG PROJDIR="/workspace/${PROJECT_NAME}"
# Make non-interactive environment.
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# Don't enable bytecode compilation
ENV UV_COMPILE_BYTECODE=0
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

## Preparing env for uv.
RUN echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc \
    && echo 'eval "$(uvx --generate-shell-completion bash)"' >> ~/.bashrc \
    && uv python install ${PYTHON_VERSION} \
    ## Installing dependencies.
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates lsb-release \
        openssh-client gnupg2 gnupg-agent \
        build-essential \
        # Database: \
        postgresql-common \
        wget \
        git \
        curl \
        graphviz\
        # non-root user: \
        sudo \
    # Postgres
    && install -d /usr/share/postgresql-common/pgdg \
    && curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc --fail https://www.postgresql.org/media/keys/ACCC4CF8.asc \
    && sh -c 'echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list' \
    && apt-get update -y \
    && apt-get install -y postgresql-${POSTGRES_VERSION} postgresql-contrib-${POSTGRES_VERSION} \    
    # docker: \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
    # non-root user: \
    && if ! getent group ${USER_GID}; then \
        groupadd --gid ${USER_GID} ${USERNAME}; \
    fi \
    && if id -u ${USER_UID} >/dev/null 2>&1; then \
        existing_user=$(getent passwd ${USER_UID} | cut -d: -f1) \
        && usermod -l ${USERNAME} -d /home/${USERNAME} -m ${existing_user}; \
    else \
        useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}; \
    fi \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    # user to docker group: \
    && usermod -aG docker ${USERNAME} \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt \
    apt-get clean

RUN go run github.com/playwright-community/playwright-go/cmd/playwright@latest install chromium --with-deps

USER $USERNAME

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR ${PROJDIR}

ENV DEVCONTAINER=true

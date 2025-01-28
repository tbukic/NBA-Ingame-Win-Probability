ARG UBUNTU_VARIANT="ubuntu22.04" 
ARG UV_VERSION="0.5.7"
ARG USER="accelerate"
ARG PYTHON_VERSION='3.12'
ARG POSTGRES_VERSION='16'
ARG PROJECT_NAME='project'
ARG PROJDIR

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_image

# Start from a CUDA development image
FROM ubuntu:${UBUNTU_VARIANT} AS build-env
COPY --from=uv_image /uv /uvx /bin/
 
ARG USER
ARG PYTHON_VERSION
ARG POSTGRES_VERSION
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
        curl \
    # Postgres
    && install -d /usr/share/postgresql-common/pgdg \
    && curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc --fail https://www.postgresql.org/media/keys/ACCC4CF8.asc \
    && sh -c 'echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list' \
    && apt-get update -y \
    && apt-get install -y postgresql-${POSTGRES_VERSION} postgresql-contrib-${POSTGRES_VERSION} \        
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt \
    apt-get clean

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR ${PROJDIR}
COPY ./.streamlit ./.streamlit
COPY ./models ./models
COPY ./nba_betting_ai ./nba_betting_ai
COPY ./.python-version ./.python-version
COPY ./pyproject.toml ./pyproject.toml
COPY ./uv.lock ./uv.lock
COPY ./README.md ./README.md

CMD ["uv", "run", "streamlit", "run", "nba_betting_ai/app.py", "--server.address=0.0.0.0"]


FROM python:3.13 AS base

WORKDIR /tmp
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -yy \
    && apt-get install -yy --no-install-suggests --no-install-recommends \
    pipx gfortran libopenblas-dev \
    libx11-dev libxcb-cursor0 qt6-base-dev

RUN --mount=type=cache,target=/root/.cache \
    pipx install poetry 
RUN --mount=type=cache,target=/root/.cache \
    pipx inject poetry \
    # poetry-plugin-bundle \
    poetry-plugin-export

WORKDIR /deps
COPY poetry.lock pyproject.toml ./
# COPY  requirements-exp.txt ./

ENV PATH=/root/.local/bin:${PATH}
RUN poetry lock
RUN poetry export --without-hashes --without-urls --all-groups > requirements-exp.txt

ENV PYTHONUNBUFFERED=1
#ENV POETRY_VIRTUALENVS_CREATE=false
#     POETRY_CACHE_DIR='/var/cache/pypoetry' \
#     POETRY_HOME='/usr/local' \
ENV PIP_BREAK_SYSTEM_PACKAGES  1

RUN --mount=type=cache,target=/root/.cache \
    pip3 install torch --index-url \
    # https://download.pytorch.org/whl/rocm6.3
    https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache \
    pip3 install -r /deps/requirements-exp.txt



WORKDIR /app

COPY . .

# RUN --mount=type=cache,target=/root/.cache \
#     pip3 install -r requirements.txt

# RUN --mount=type=cache,target=/root/.cache \
#     --mount=type=cache,target=/var/cache/pypoetry \
#     poetry install -v --no-interaction --no-ansi

# RUN poetry bundle venv --no-interaction --no-ansi --python=/usr/bin/python3 /usr/local

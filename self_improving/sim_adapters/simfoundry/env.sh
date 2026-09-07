#!/usr/bin/env bash
# Project-local runtime; source this before upstream installation or pipeline commands.
SF_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export MAMBA_ROOT_PREFIX="${SF_WORKSPACE}/.cache/simfoundry/miniforge"
export CONDA_ENVS_PATH="${MAMBA_ROOT_PREFIX}/envs"
export CONDA_PKGS_DIRS="${MAMBA_ROOT_PREFIX}/pkgs"
export CONDARC="${SF_WORKSPACE}/.cache/simfoundry/condarc"
export HY3DGEN_MODELS="${SF_WORKSPACE}/.cache/simfoundry/hy3dgen"
export HF_HUB_CACHE="${SF_WORKSPACE}/.cache/simfoundry/hub"
export HF_XET_CACHE="${SF_WORKSPACE}/.cache/simfoundry/xet"
export HF_TOKEN_PATH="${SF_WORKSPACE}/.cache/simfoundry/hf_token"
export PIP_CACHE_DIR="${SF_WORKSPACE}/.cache/simfoundry/pip"
export PATH="${MAMBA_ROOT_PREFIX}/bin:${PATH}"
export MAX_JOBS="${MAX_JOBS:-8}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset SF_WORKSPACE

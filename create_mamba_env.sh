#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# should match env name from YAML
ENV_NAME=omni_robo_gym

pushd "${ROOT_DIR}/"

    # setup mamba
    MAMBA_DIR="$(mamba info --base)"
    
    source "${MAMBA_DIR}/etc/profile.d/mamba.sh"

    # !!! this removes existing version of the env
    mamba remove -y -n "${ENV_NAME}" --all

    # create the env from YAML
    mamba env create -f ./omni_robo_gym.yml

    # activate env
    # mamba activate "${ENV_NAME}"

    # # install omni_robo_gym package in editable mode
    # pip install -e .

popd
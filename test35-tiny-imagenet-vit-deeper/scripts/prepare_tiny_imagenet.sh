#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?usage: prepare_tiny_imagenet.sh ROOT}
DATA_DIR="${ROOT}/data"
SRC_ROOT="/data/run01/scwb923/4.3-log/test30-tiny-imagenet-vit/data"

mkdir -p "${DATA_DIR}"

if [ ! -e "${DATA_DIR}/tiny-imagenet-200" ] && [ ! -e "${DATA_DIR}/tiny-imagenet-200.zip" ]; then
  if [ -e "${SRC_ROOT}/tiny-imagenet-200" ]; then
    cp -al "${SRC_ROOT}/tiny-imagenet-200" "${DATA_DIR}/tiny-imagenet-200"
  elif [ -e "${SRC_ROOT}/tiny-imagenet-200.zip" ]; then
    cp -f "${SRC_ROOT}/tiny-imagenet-200.zip" "${DATA_DIR}/tiny-imagenet-200.zip"
  else
    echo "Tiny-ImageNet source data not found under ${SRC_ROOT}" >&2
    exit 1
  fi
fi

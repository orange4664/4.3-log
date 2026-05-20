#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?usage: prepare_tiny_imagenet.sh <root>}
DATA_DIR="${ROOT}/data"
ARCHIVE="${DATA_DIR}/tiny-imagenet-200.zip"
UNPACKED="${DATA_DIR}/tiny-imagenet-200"
URL="https://cs231n.stanford.edu/tiny-imagenet-200.zip"

mkdir -p "${DATA_DIR}"

if [ -d "${UNPACKED}" ]; then
  echo "Tiny-ImageNet already prepared at ${UNPACKED}"
  exit 0
fi

if [ ! -f "${ARCHIVE}" ]; then
  wget -O "${ARCHIVE}" "${URL}"
fi

unzip -q -o "${ARCHIVE}" -d "${DATA_DIR}"
echo "Prepared Tiny-ImageNet at ${UNPACKED}"

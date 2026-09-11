#!/usr/bin/env bash

# 该脚本只负责定位项目目录并转发参数，参数解析统一由 options.py 完成。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" proxyfl.py "$@"

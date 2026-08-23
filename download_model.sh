#!/bin/bash
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/model.gguf"

MODEL_URL="https://huggingface.co/EYEDOL/agri-advisor-1.5b/resolve/main/agri-Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"
echo "downloading $MODEL_URL -> $MODEL_FILE ..."
curl -L -o "$MODEL_FILE" "$MODEL_URL"
echo "done."

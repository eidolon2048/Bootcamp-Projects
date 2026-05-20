#!/bin/zsh

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_BASE="/Users/roma/anaconda3"
ENV_NAME="streamlit_environment"

if [ ! -f "$CONDA_BASE/bin/activate" ]; then
  echo "Anaconda activate script not found at $CONDA_BASE/bin/activate"
  echo "Press any key to close this window."
  read -k 1
  exit 1
fi

source "$CONDA_BASE/bin/activate"
conda activate "$ENV_NAME"
cd "$APP_DIR"

echo "Starting Emotion Recognition Streamlit App..."
echo "App folder: $APP_DIR"
echo "Conda env: $ENV_NAME"
echo

streamlit run app.py

echo
echo "Streamlit stopped. Press any key to close this window."
read -k 1

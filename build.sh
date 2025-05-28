#!/bin/sh
cd `dirname $0`

eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# Set specific Python version
PYENV_VERSION="3.13.3"

python3 -m venv .build-env
source .build-env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt


# Create a virtual environment to run our code
VENV_NAME="venv"
PYTHON="$VENV_NAME/bin/python"

if ! $PYTHON -m pip install pyinstaller -Uqq; then
    exit 1
fi

$PYTHON -m PyInstaller --onefile --hidden-import="googleapiclient" src/main.py
tar -czvf dist/archive.tar.gz ./dist/main


#!/bin/bash
cd `dirname $0`

# Desired version of Python to build with
PYTHON_VERSION="3.13.3"

# For uv (if available)
if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env"
fi

# Check if uv is installed and use it for faster package management
if command -v uv >/dev/null; then
    echo "uv found. Creating virtual environment and installing packages..."
    if ! uv venv --python $PYTHON_VERSION; then
        echo "Failed to create virtual environment with Python $PYTHON_VERSION"
        exit 1
    fi

    if ! uv pip install -r requirements.txt; then
        echo "Failed to install packages"
        exit 1
    fi
else
    # Fallback to standard venv approach
    VENV_NAME="venv"
    PYTHON="$VENV_NAME/bin/python"
    ENV_ERROR="This module requires Python $PYTHON_VERSION, pip, and virtualenv to be installed."
    
    # Try to use python3.13 specifically
    if ! python3.13 -m venv $VENV_NAME >/dev/null 2>&1; then
        echo "Failed to create virtualenv with python3.13."
        if command -v apt-get >/dev/null; then
            echo "Detected Debian/Ubuntu, attempting to install python3.13-venv automatically."
            SUDO="sudo"
            if ! command -v $SUDO >/dev/null; then
                SUDO=""
            fi
            if ! apt info python3.13-venv >/dev/null 2>&1; then
                echo "Package info not found, trying apt update"
                $SUDO apt -qq update >/dev/null
            fi
            $SUDO apt install -qqy python3.13-venv >/dev/null 2>&1
            if ! python3.13 -m venv $VENV_NAME >/dev/null 2>&1; then
                echo $ENV_ERROR >&2
                exit 1
            fi
        else
            echo $ENV_ERROR >&2
            exit 1
        fi
    fi

    echo "Virtualenv found/created. Installing/upgrading Python packages..."
    if ! [ -f .installed ]; then
        if ! $PYTHON -m pip install -r requirements.txt -Uqq; then
            exit 1
        else
            touch .installed
        fi
    fi
fi

source .venv/bin/activate
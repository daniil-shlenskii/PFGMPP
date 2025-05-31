SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_DIR=$SCRIPT_DIR
export SRC_DIR=${PROJECT_DIR}/src

export PYTHONPATH=${SRC_DIR}:${PYTHONPATH}

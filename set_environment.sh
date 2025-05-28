SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_PATH=$SCRIPT_DIR
export SRC_DIR=${PROJECT_PATH}/src

export PYTHONPATH=${SRC_DIR}:${PYTHONPATH}

#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${DIR}/../../../:${PYTHONPATH}
export PYTHONPATH=${DIR}/../../../slim:${PYTHONPATH}
export PYTHONPATH=${DIR}/../../../object_detection:${PYTHONPATH}
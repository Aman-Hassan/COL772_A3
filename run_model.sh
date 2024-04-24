#!/bin/bash

if [ "$1" == "test" ]; then
    # Number of default parameters - 1, Number of user-defined parameters - 3
    if [ "$#" -ne 4 ]; then
        echo "Error: Illegal number of parameters. Please use the format: bash run_model.sh test <path_to_data> <path_to_model> <path_to_result"
        exit 2
    fi
    # echo "Running tests..."
    python3 A3_v1.py "test" "$2" "$3" "$4"
else if [ "$1" == "train" ]; then
    # Number of default parameters - 1, Number of user-defined parameters - 2
    if [ "$#" -ne 3 ]; then
        echo "Error: Illegal number of parameters. Please use the format: bash run_model.sh <path_to_data> <path_to_save>"
        exit 2
    fi
    # echo "Training..."
    python3 A3_v1.py "train" "$2" "$3"
else
    echo "Error: Invalid argument. Please use either 'train' or 'test' as the first argument."
    exit 2
fi
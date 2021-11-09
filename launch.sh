#!/bin/bash

syntax_error(){
    echo "Syntax: sh launch.sh start|build postgres|rtos-cpu|rtos-gpu"
}

if [ "$1" = "start" -a "$2" = "postgres" ]; then
    docker-compose up -d postgres;
    if [ "$3" = "init" ]; then
        docker exec -i postgres pg_restore -U postgres -x --no-privileges --no-owner -Fc -d imdbload < $4;
    else
        echo "Unknown command $3";
        exit 1;
    fi
elif [ "$1" = "start" -a "$2" = "rtos-cpu" ]; then
    docker run -it --rm \
        -e RTOS_JOB_DIR="JOB-queries" \
        -e RTOS_SCHEMA_FILE="schema.sql" \
        -e RTOS_DB_PASSWORD="123456" \
        -e RTOS_DB_USER="postgres" \
        -e RTOS_DB_NAME="imdbload" \
        -e RTOS_DB_HOST="0.0.0.0" \
        -e RTOS_DB_PORT="5432" \
        --network host \
        -v $(realpath ./models):/workplace/models \
        -p '5432:5432' \
        ai4dbcode_rtos
elif [ "$1" = "start" -a "$2" = "rtos-gpu" ]; then
    docker run -it --rm \
        --gpus all \
        --device /dev/nvidia0 --device /dev/nvidia-modeset \
        --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
        --device /dev/nvidiactl \
        -e RTOS_JOB_DIR="JOB-queries" \
        -e RTOS_SCHEMA_FILE="schema.sql" \
        -e RTOS_DB_PASSWORD="123456" \
        -e RTOS_DB_USER="postgres" \
        -e RTOS_DB_NAME="imdbload" \
        -e RTOS_DB_HOST="0.0.0.0" \
        -e RTOS_DB_PORT="5432" \
        --network host \
        -v $(realpath ./models):/workplace/models \
        -p '5432:5432' \
        ai4dbcode_rtos
elif [ "$1" = build ]; then
    if [ "$2" = "postgres" -o "$2" = "rtos-cpu" -o "$2" = "rtos-gpu" ]; then
        target=$( echo "$2" | egrep -o '^[a-z0-9]+')
        docker-compose build $target
    else
        echo "Cannot build unknown target $2";
        exit 1;
    fi
else
    syntax_error;
fi
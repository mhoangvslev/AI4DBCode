#!/bin/bash

syntax_error(){
    echo "Syntax: sh launch.sh start|build postgres|rtos-cpu|rtos-gpu"
}

if [ "$1" = "start" -a "$2" = "postgres" ]; then
    docker-compose up -d postgres;
    if [ "$3" = "init" ]; then
        docker exec -i postgres -Fc pg_restore -U postgres -x --no-privileges --no-owner -d imdbload < $4;
    else
        echo "Unknown command $3";
        exit -1;
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
        -p '5432:5432' \
        jos_rtos-cpu
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
        -e RTOS_DB_PORT="5433" \
        --network postgresNetwork \
        -p '5433:5432' \
        jos_rtos-gpu
elif [ "$1" = build ]; then
    if [ "$2" = "postgres" -o "$2" = "rtos-cpu" -o "$2" = "rtos-gpu" ]; then
        docker-compose build $2
    else
        echo "Cannot build unknown target $2";
        exit -1;
    fi
else
    syntax_error;
fi
#!/bin/bash

syntax_error(){
    echo "Syntax: 
    (1) Setup: sh launch.sh <start|build> <postgres|rtos-cpu|rtos-gpu>
    (2) Train: sh launch.sh <cost-training|latency-tuning|train> <sql|sparql> [-debug]
    "
}

if [ "$1" = "start" -a "$2" = "postgres" ]; then
    docker-compose up -d postgres;
    if [ "$3" = "init" ]; then
        docker exec -i postgres pg_restore -U postgres -x --no-privileges --no-owner -Fc -d imdbload < $4;
        exit 0
    else
        echo "Unknown command $3";
        exit 1;
    fi
elif [ "$1" = "start" -a "$2" = "virtuoso" ]; then
    VIRTUOSO_DB=$VIRTUOSO_DB docker-compose up -d virtuoso;
    if docker exec virtuoso bash -c 'echo "sparql select distinct ?g where { graph ?g { ?s a ?c } };" > tmp.sparql && ./isql localhost:1111 dba dba tmp.sparql' | grep -o "http://example.com/DAV/void" ; then
        echo "Virtuoso launched sucessfully!"
        exit 0
    else
        echo "Virtuoso did not launched successfully. It could be:
            (1) You must specify where to look for virtuoso database folder in VIRTUOSO_DB
            (2) Unknown error
        "
        exit 1
    fi

elif [ "$1" = "start" -a "$2" = "rtos-cpu" ]; then
    docker run -it --rm \
        -e RTOS_JOB_DIR="JOB-queries/$3$4" \
        -e RTOS_SCHEMA_FILE="schema.sql" \
        -e RTOS_DB_PASSWORD="123456" \
        -e RTOS_DB_USER="postgres" \
        -e RTOS_DB_NAME="imdbload" \
        -e RTOS_DB_HOST="0.0.0.0" \
        -e RTOS_DB_PORT="5432" \
        -e RTOS_ISQL_ENDPOINT="sparql" \
        -e RTOS_ISQL_GRAPH="http://example.com/DAV/void" \
        -e RTOS_ISQL_HOST="localhost" \
        -e RTOS_ISQL_PORT="8890" \
        -e RTOS_PYTORCH_DEVICE="gpu" \
        -e RTOS_JTREE_BUSHY=0 \
        -e RTOS_ENGINE="$3" \
        --network host \
        -v $(realpath ./models):/workplace/models \
        -v $(realpath ./log):/workplace/log \
        -v /tmp:/tmp \
        -p '5432:5432' \
        -p '4000:4000' \
        ai4dbcode-rtos_rtos
elif [ "$1" = "start" -a "$2" = "rtos-gpu" ]; then
    docker run -it --rm \
        --gpus all \
        --device /dev/nvidia0 --device /dev/nvidia-modeset \
        --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
        --device /dev/nvidiactl \
        -e RTOS_JOB_DIR="JOB-queries/$3$4" \
        -e RTOS_SCHEMA_FILE="schema.sql" \
        -e RTOS_DB_PASSWORD="123456" \
        -e RTOS_DB_USER="postgres" \
        -e RTOS_DB_NAME="imdbload" \
        -e RTOS_DB_HOST="0.0.0.0" \
        -e RTOS_DB_PORT="5432" \
        -e RTOS_ISQL_ENDPOINT="sparql" \
        -e RTOS_ISQL_GRAPH="http://example.com/DAV/void" \
        -e RTOS_ISQL_HOST="localhost" \
        -e RTOS_ISQL_PORT="8890" \
        -e RTOS_PYTORCH_DEVICE="gpu" \
        -e RTOS_JTREE_BUSHY=0 \
        -e RTOS_ENGINE="$3" \
        --network host \
        -v $(realpath ./models):/workplace/models \
        -v $(realpath ./log):/workplace/log \
        -v /tmp:/tmp \
        -p '5432:5432' \
        -p '4000:4000' \
        ai4dbcode-rtos_rtos
elif [ "$1" = "build" ]; then
    if [ "$2" = "postgres" -o "$2" = "rtos-cpu" -o "$2" = "rtos-gpu" ]; then
        docker-compose build $( echo "$2" | egrep -o '^[a-z0-9]+')
    else
        echo "Cannot build unknown target $2";
        exit 1;
    fi
elif [ "$1" = "cost-training" ]; then
    find JOB-queries/$2$3 -mindepth 1 -type d -exec rm -rf '{}' \;
    find JOB-queries/$2$3/*.csv -type f -exec rm '{}' \;
    RTOS_JOB_DIR="JOB-queries/$2$3" \
    RTOS_SCHEMA_FILE="schema.sql" \
    RTOS_DB_PASSWORD="123456" \
    RTOS_DB_USER="postgres" \
    RTOS_DB_NAME="imdbload" \
    RTOS_DB_HOST="0.0.0.0" \
    RTOS_DB_PORT="5432" \
    RTOS_ISQL_ENDPOINT="sparql" \
    RTOS_ISQL_GRAPH="http://example.com/DAV/void" \
    RTOS_ISQL_HOST="localhost" RTOS_ISQL_PORT="8890" \
    RTOS_JTREE_BUSHY=0 \
    RTOS_PYTORCH_DEVICE="cpu" \
    RTOS_GV_FORMAT="png" \
    RTOS_ENGINE="$2" \
    python CostTraining.py
elif [ "$1" = "latency-tuning" ]; then
    find JOB-queries/$2$3 -mindepth 1 -type d -exec rm -rf '{}' \;
    find JOB-queries/$2$3/*.csv -type f -exec rm '{}' \;
    RTOS_JOB_DIR="JOB-queries/$2$3" \
    RTOS_SCHEMA_FILE="schema.sql" \
    RTOS_DB_PASSWORD="123456" \
    RTOS_DB_USER="postgres" \
    RTOS_DB_NAME="imdbload" \
    RTOS_DB_HOST="0.0.0.0" \
    RTOS_DB_PORT="5432" \
    RTOS_ISQL_ENDPOINT="sparql" \
    RTOS_ISQL_GRAPH="http://example.com/DAV/void" \
    RTOS_ISQL_HOST="localhost" RTOS_ISQL_PORT="8890" \
    RTOS_JTREE_BUSHY=0 \
    RTOS_PYTORCH_DEVICE="cpu" \
    RTOS_GV_FORMAT="png" \
    RTOS_ENGINE="$2" \
    python LatencyTuning.py
elif [ "$1" = "train" ]; then
    find JOB-queries/$2$3 -mindepth 1 -type d -exec rm -rf '{}' \;
    find JOB-queries/$2$3/*.csv -type f -exec rm '{}' \;
    RTOS_JOB_DIR="JOB-queries/$2$3" \
    RTOS_SCHEMA_FILE="schema.sql" \
    RTOS_DB_PASSWORD="123456" \
    RTOS_DB_USER="postgres" \
    RTOS_DB_NAME="imdbload" \
    RTOS_DB_HOST="0.0.0.0" \
    RTOS_DB_PORT="5432" \
    RTOS_ISQL_ENDPOINT="sparql" \
    RTOS_ISQL_GRAPH="http://example.com/DAV/void" \
    RTOS_ISQL_HOST="localhost" RTOS_ISQL_PORT="8890" \
    RTOS_JTREE_BUSHY=0 \
    RTOS_PYTORCH_DEVICE="cpu" \
    RTOS_GV_FORMAT="png" \
    RTOS_ENGINE="$2" \
    python train.py
else
    syntax_error;
fi

#!/bin/bash

syntax_error(){
    echo "Syntax: 
    (1) Setup: sh launch.sh <start|build> <postgres|rtos-cpu|rtos-gpu>
    (2) Train: [DEBUG='-debug'] sh launch.sh <cost-training|latency-tuning|train> <train|predict> <sql|sparql> <cost|cost-improvement|rtos|foop-cost> n_episodes
    "
}

export RTOS_SCHEMA_FILE="schema.sql"
export RTOS_DB_PASSWORD="123456"
export RTOS_DB_USER="postgres"
export RTOS_DB_NAME="imdbload"
export RTOS_DB_HOST="0.0.0.0"
export RTOS_DB_PORT="5432"
export RTOS_ISQL_ENDPOINT="sparql"
export RTOS_ISQL_GRAPH="http://example.com/DAV/void"
export RTOS_ISQL_HOST="localhost" RTOS_ISQL_PORT="8890"
export RTOS_JTREE_BUSHY=0
export RTOS_PYTORCH_DEVICE="gpu"
export RTOS_GV_FORMAT="png"
export RTOS_ENGINE="$3"
export RTOS_JOB_DIR="JOB-queries/$3$DEBUG"
export RTOS_SPARQL_EXTRA_FEAT="no"

if [ "$1" = "start" -a "$2" = "postgres" ]; then
    docker-compose up -d postgres;
    if [ "$3" = "init" ]; then
        docker exec -i postgres pg_restore -U postgres -x --no-privileges --no-owner -Fc -d imdbload < $4;
        exit 0
    elif [ ! -z "$3" ]; then
        echo "Unknown command $3";
        exit 1;
    fi
    exit 0;
elif [ "$1" = "start" -a "$2" = "virtuoso" ]; then
    VIRTUOSO_DB=$VIRTUOSO_DB docker-compose up -d virtuoso;
    attempt=0

    until echo $(docker exec virtuoso bash -c 'echo "sparql select distinct ?g where { graph ?g { ?s a ?c } };" > tmp.sparql && ./isql localhost:1111 dba dba tmp.sparql') | grep -o "http://example.com/DAV/void" ; 
    do
        attempt=$(expr $attempt + 1)
        echo "Making attempt #$attempt...";
        sleep 1;

        echo $test

        if [ "$attempt" = "$3" ]; then
            echo "Virtuoso did not launched successfully. It could be:
                (1) You must specify where to look for virtuoso database folder in VIRTUOSO_DB
                (2) Check log below
            $test
            "
            exit 1
        fi
    done

    echo "Virtuoso is succesfully setup!"
    
elif [ "$1" = "start" -a "$2" = "rtos-cpu" ]; then
    docker run -it --rm \
        -e RTOS_JOB_DIR="JOB-queries/$3" \
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
        -e RTOS_JOB_DIR="JOB-queries/$3" \
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
    rm -f *.log
    if [ "$2" = "train" ]; then
        find JOB-queries/$3$DEBUG -mindepth 1 -type d -exec rm -rf '{}' \;
        find JOB-queries/$3$DEBUG/*.csv -type f -exec rm '{}' \;
        RTOS_JOB_DIR="JOB-queries/$3$DEBUG" \
            python CostTraining.py --mode "train" --reward "$4" --n_episodes $5 --queryfile "$6" 
    elif [ "$2" = "predict" ]; then
        python CostTraining.py --log_level "DEBUG" --mode "predict" --reward "$4" --queryfile "$5" 
    fi

elif [ "$1" = "latency-tuning" ]; then
    rm -f *.log
    if [ "$2" = "train" ]; then
        find JOB-queries/$3$DEBUG -mindepth 1 -type d -exec rm -rf '{}' \;
        find JOB-queries/$3$DEBUG/*.csv -type f -exec rm '{}' \;
        RTOS_JOB_DIR="JOB-queries/$3$DEBUG" \
            python LatencyTuning.py --mode "train" --reward "$4" --n_episodes $5 --queryfile "$6" 
    elif [ "$2" = "predict" ]; then
        python LatencyTuning.py --mode "predict" --reward "$4" --queryfile "$5" 
    fi

elif [ "$1" = "train" ]; then 
    rm -f *.log
    if [ "$2" = "train" ]; then
        find JOB-queries/$3$DEBUG -mindepth 1 -type d -exec rm -rf '{}' \;
        find JOB-queries/$3$DEBUG/*.csv -type f -exec rm '{}' \;
        python train.py --mode "train" --reward "$4" --n_episodes $5 --queryfile "$6" 
    elif [ "$2" = "predict" ]; then
        python train.py --mode "predict" --reward "$4" --queryfile "$5" 
    fi
else
    syntax_error;
fi

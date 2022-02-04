#!/bin/bash

syntax_error(){
    echo "Syntax: 
    (1) Setup: sh launch.sh <start|build> <postgres|rtos-cpu|rtos-gpu>
    (2) Train: [DEBUG='-debug'] sh launch.sh <cost-training|latency-tuning|train> <train|predict> <sql|sparql> <cost|cost-improvement|rtos|foop-cost> n_episodes
    "
}

if [ -z $VIRTUOSO_DB ]; then
    echo "VIRTUOSO_DB is not set, defaulting to current directory..."
    export VIRTUOSO_DB=$(realpath .)
fi

if [ -z $SAGE_CONFIG ]; then
    echo "SAGE_CONFIG is not set, defaulting to current directory..."
    export SAGE_CONFIG=$(realpath .)
fi

if [ -z $SAGE_GRAPH ]; then
    echo "SAGE_GRAPH is not set, defaulting to current directory..."
    export SAGE_GRAPH=$(realpath .)
fi

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
    export VIRTUOSO_DB=$(realpath $VIRTUOSO_DB)
    docker-compose up -d virtuoso;
    attempt=0

    until echo $(docker exec virtuoso bash -c 'echo "sparql select distinct ?g where { graph ?g { ?s a ?c } };" > tmp.sparql && ./isql localhost:1111 dba dba tmp.sparql') | grep -o "http://example.com/DAV/void" ; 
    do
        attempt=$(expr $attempt + 1)
        echo "Making attempt #$attempt...";
        sleep 1;

        if [ "$attempt" = "$3" ]; then
            echo "Virtuoso did not launched successfully. It could be:
                (1) You must specify where to look for virtuoso database folder in VIRTUOSO_DB
                (2) Check error message
            "
            exit 1
        fi
    done

    echo "Virtuoso is succesfully setup!"

elif [ "$1" = "start" -a "$2" = "sage" ]; then
    export SAGE_CONFIG=$(realpath $SAGE_CONFIG) 
    export SAGE_GRAPH=$(realpath $SAGE_GRAPH) 
    docker-compose up -d sage-engine;
    attempt=0

    until echo $(curl -H "Content-Type: application/json" -d '{"query": "SELECT * WHERE { ?s a ?c }", "defaultGraph": "http://localhost:8081/sparql/jobrdf"}' http://localhost:8081/sparql) | grep -o "cost" ; 
    do
        attempt=$(expr $attempt + 1)
        echo "Making attempt #$attempt...";
        sleep 1;

        if [ "$attempt" = "$3" ]; then
            echo "Virtuoso did not launched successfully. It could be:
                (1) You must specify where to look for virtuoso database folder in VIRTUOSO_DB
                (2) Check error message
            "
            exit 1
        fi
    done

    echo "SaGe engine is succesfully setup!"

elif [ "$1" = "start" -a "$2" = "jupyter" ]; then
    docker-compose build jupyter-notebook;
    docker run -it --rm \
        -v $(realpath ./):/workplace/ \
        -v /tmp:/tmp \
        -p '8080:8080' \
        jupyter/scipy-notebook bash -c 'jupyter notebook --allow-root --NotebookApp.allow_origin=https://colab.research.google.com --port=8080 --NotebookApp.port_retries=0'
    exit 0;
    
elif [ "$1" = "start" -a "$2" = "rtos-cpu" ]; then
    docker run -it --rm \
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
    export RTOS_TRAINTYPE="cost-training"
    shift;
    python CostTraining.py $*

elif [ "$1" = "latency-tuning" ]; then
    rm -f *.log
    export RTOS_TRAINTYPE="latency-tuning"
    shift;
    python LatencyTuning.py $*

elif [ "$1" = "train" ]; then 
    rm -f *.log
    export RTOS_TRAINTYPE="train"
    shift;
    python train.py $*
else
    syntax_error;
fi

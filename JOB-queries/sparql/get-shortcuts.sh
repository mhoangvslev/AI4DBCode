#!/bin/bash

#awk -F"\n" '/Shortcuts/{while( $0 !~ "Attribute selection clauses" ){getline;print;}}' JOB-queries/sparql/* \

if [ "$1" = "shortcut" ]; then
    sed -n '/Shortcuts/,/2/{p;/^Attribute selection pattern/q}' $(realpath $2)/*.sparql \
        | grep -Po '<[\w\:\/\.\#]+>' \
        | sed "s/[<>]//g" \
        | sort \
        | uniq
elif [ "$1" = "relations" ]; then
    cat $(realpath $2)/*.sparql \
        | grep -Po '<[\w\:\/\.\#]+>' \
        | sed "s/[<>]//g" \
        | sort \
        | uniq 

elif [ "$1" = "columns" ]; then
    cat $(realpath $2)/*.sparql \
        | grep -Po '\?\w+' \
        | sed "s/[<>]//g" \
        | sort \
        | uniq 
fi
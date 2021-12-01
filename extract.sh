#!/bin/bash

syntax_error(){
    echo "Syntax error:"
    echo "sh extract.sh <relations|shortcuts|columns> path/to/workload"
}

if [ "$1" = "shortcuts" ]; then
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
else
    syntax_error;
fi
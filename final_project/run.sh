#!/bin/bash

filename=

while [ -n "$1" ]; do
    case $1 in 
        -i  | --file)               shift
                                    filename=$1
                                    ;;
        -h  | --help)               echo "# Usage: -i [file]"
                                    exit
                                    ;;
        * )                         echo "Option $1 not recognized!"
    esac
    shift
done

./exe ${filename}
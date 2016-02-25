#!/usr/bin/env bash

printError()
{
    echo -e "\033[0;31mERROR: $1\033[0m" >&2
}

errorAndExit()
{
    printError "$1"
    exit 2
}

getNameFromFolder()
{
    local exampleDir="$1"
    local example_name=`basename "$exampleDir"`
    #if we only have one case we must skip folder with name . and read real folder name
    if [ "$example_name" == "." ] ; then
        exampleDir=`dirname "$exampleDir"`
        example_name=`basename "$exampleDir"`
    fi
    echo "$example_name"
}

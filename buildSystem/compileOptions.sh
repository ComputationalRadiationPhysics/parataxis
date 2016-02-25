#!/usr/bin/env bash

# Prints the help. Requires 'exeDescription' to be set
helpCompile()
{
    echo $exeDescription
    echo ""
    echo "usage: $0 [OPTION] src_dir dest_dir"
    echo ""
    echo "-l                   - interpret all folders in src_dir as examples and process each of it"
    echo "-j <N>               - spawn N tests in parallel"
    echo "-h | --help          - show this help message"
    echo ""
}

# Parses and validates the params passed
# Sets: isExampleList=0|1
#       numParallel=1-n
#       srcDir, destinationDir
#       list_examples=<list of folders>
parseCompileOptions()
{

    # options may be followed by one colon to indicate they have a required argument
    OPTS=`getopt -o lj:h -l help -- "$@"`
    if [ $? != 0 ] ; then
        # something went wrong, getopt will put out an error message for us
        exit 1
    fi

    eval set -- "$OPTS"

    isExampleList=0
    numParallel=1

    while true ; do
        case "$1" in
            -h|--help)
                helpCompile
                exit 1
                ;;
            -j)
                numParallel="$2"
                shift
                ;;
            -l)
                isExampleList=1
                ;;
            --) shift; break;;
        esac
        shift
    done

    if [ $# -ne 2 ]; then
        printError "Missing src and/or destination directory."
        helpCompile
        exit 1
    fi

    srcDir=$1
    destinationDir=$2

    if [ $isExampleList -ne 1 ]; then
        # single test
        list_examples="."
    else
        # examples/ folder
        list_examples=`ls -w1 $srcDir`
    fi
    
}


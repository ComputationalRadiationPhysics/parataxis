#!/usr/bin/env bash

# Prints the help. Requires 'exeDescription' to be set
helpTests()
{
    echo $exeDescription
    echo ""
    echo "usage: $0 [OPTION] -e src_dir -o dest_dir -- [TEST_OPTIONS]"
    echo ""
    echo "-e src_dir           - path to example or examples (with -l)"
    echo "-o dest_dir          - (base-)path where the example(s) was/will be build"
    echo "-l                   - interpret all folders in src_dir as examples and process each of it"
    echo "-h | --help          - show this help message"
    echo ""
    echo "[TEST_OPTIONS] will be directly passed to the execTests.py script"
    echo ""
}

# Parses and validates the params passed
# Sets: isExampleList=0|1
#       srcDir, destinationDir
#       list_examples=<list of folders>
#       testOptions=<additional options>
parseTestOptions()
{

    # options may be followed by one colon to indicate they have a required argument
    OPTS=`getopt -o lhe:o: -l help -- "$@"`
    if [ $? != 0 ] ; then
        # something went wrong, getopt will put out an error message for us
        exit 1
    fi

    eval set -- "$OPTS"

    isExampleList=0
    srcDir=""
    destinationDir=""

    while true ; do
        case "$1" in
            -h|--help)
                helpTests
                exit 1
                ;;
            -l)
                isExampleList=1
                ;;
            -e)
                srcDir=${2%/}
                shift
                ;;
            -o)
                destinationDir=${2%/}
                shift
                ;;
            --) shift; break;;
        esac
        shift
    done
    
    if [ "$srcDir" == "" ] || [ "$destinationDir" == "" ]; then
        printError "Missing src and/or destination directory."
        helpTests
        exit 1
    fi
    
    testOptions="$@"

    if [ $isExampleList -ne 1 ]; then
        # single test
        list_examples="."
    else
        # examples/ folder
        list_examples=`ls -w1 $srcDir`
    fi
    
}


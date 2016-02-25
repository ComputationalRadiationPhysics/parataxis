#!/usr/bin/env bash

this_dir="$( cd "$(dirname "$0")" ; pwd )"

help()
{
    echo "Configures and compiles a specific example set"
    echo ""
    echo "usage: $0 [OPTIONS] <example name/directory> <build directory>"
    echo "-h | --help   - Show help"
    echo "-t<num>       - Use cmake preset <num>"
    echo "-f            - Force overwrite of build directory"
    echo ""
}

printError()
{
    echo -e "\033[0;31mERROR: $1\033[0m" >&2
}

errorAndExit()
{
    printError "$1"
    exit 2
}

# options may be followed by one colon to indicate they have a required argument
OPTS=`getopt -o ht:f -l help -- "$@"`
if [ $? != 0 ] ; then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

eval set -- "$OPTS"

force_param=""
cmakeFlagsNr=0

while true ; do
    case "$1" in
        -h|--help)
            echo -e "$(help)"
            exit 1
            ;;
        -t)
            cmakeFlagsNr="$2"
            shift
            ;;
        -f)
            force_param="-f"
            ;;
        --) shift; break;;
    esac
    shift
done

if [ $# -ne 2 ] ; then
    printError "Missing example and/or destination directory."
    help
    exit 1
fi

example=$1
destinationDir=$2

echo "Configuring..."
$this_dir/configureExample.sh -t$cmakeFlagsNr $force_param $example $destinationDir

if [ $? -ne 0 ]; then
    result=$?
    echo $result > "$destinationDir/returnCode"
    exit $result
fi

echo "Building in $destinationDir..."
cd "$destinationDir"
make install
result=$?
cd -

echo "Write result $destinationDir/returnCode"

echo $result > "$destinationDir/returnCode"
exit $result


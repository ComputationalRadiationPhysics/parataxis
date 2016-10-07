#!/usr/bin/env bash
#
# Copyright 2015-2016 Alexander Grund
#
# This file is part of ParaTAXIS.
#
# ParaTAXIS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ParaTAXIS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.

this_dir="$( cd "$(dirname "$0")" ; pwd )"

help()
{
    echo "Creates a new folder for an example and configures it till it is ready for a compile via make install"
    echo ""
    echo "usage: $0 [OPTIONS] <example name/directory> <build directory>"
    echo "-h | --help   - Show help"
    echo "-t <num>      - Use cmake preset <num>"
    echo "-c <param>    - Parameter to pass to cmake (e.g. '-DCMAKE_BUILD_TYPE=Release')"
    echo "                Can be used multiple times"
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
OPTS=`getopt -o ht:fj:c: -l help -- "$@"`
if [ $? != 0 ] ; then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

eval set -- "$OPTS"

force_param=0
numParallel=1
cmakeFlagsNr=-1
cmakeFlags=""

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
            force_param=1                       
            ;;
        -c)
            cmakeFlags="$cmakeFlags $2"
            shift
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

example=${1%/}
destinationDir=${2%/}

if [ ! -d "$example" ]; then
    exampleDir="$this_dir/examples/$example"
    if [ ! -d "$exampleDir" ]; then
        errorAndExit "Example $example not found!"
    fi
else
    exampleDir=$example
fi

exampleDir="$( cd "$example" ; pwd )"

if [ ! -d "$destinationDir" ]; then
    mkdir -p "$destinationDir"
elif [ $force_param -eq 0 ]; then
    errorAndExit "Destination dir $destinationDir exists, use -f to overwrite!"
fi

if [ ! -d "$destinationDir" ]; then
    errorAndExit "Destination dir $destinationDir could not be created!"
fi

if [ -f "$exampleDir/cmakeFlags" ] && [ $cmakeFlagsNr -ge 0 ]; then
    cmakeFlagsDefault=`$exampleDir/cmakeFlags $cmakeFlagsNr`
    cmakeFlags="$cmakeFlagsDefault $cmakeFlags"
fi

cd $destinationDir

cmake_command="cmake $cmakeFlags -DXRT_EXTENSION_PATH=$exampleDir $this_dir"
echo -e "\033[32mcmake command:\033[0m $cmake_command"
$cmake_command
result=$?
if [ $result -eq 0 ]; then
    make clean
fi

cd -

exit $result


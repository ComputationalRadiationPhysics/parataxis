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
    echo "Configures and compiles a specific example set"
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

params="$@"

# options may be followed by one colon to indicate they have a required argument
OPTS=`getopt -o ht:c:f -l help -- "$@"`
if [ $? != 0 ] ; then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

eval set -- "$OPTS"

while true ; do
    case "$1" in
        -h|--help)
            echo -e "$(help)"
            exit 1
            ;;
        -t)
            shift
            ;;
        -c)
            shift
            ;;
        -f)
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

echo "Configuring..."
$this_dir/configureExample.sh $params

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


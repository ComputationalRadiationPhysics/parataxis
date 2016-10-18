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
#

printError()
{
    echo_r "ERROR: $1" >&2
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

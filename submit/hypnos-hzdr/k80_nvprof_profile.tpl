#!/usr/bin/env bash
# Copyright 2013-2016 Axel Huebl, Anton Helm, Rene Widera, Alexander Grund
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

# PIConGPU batch script for hypnos PBS batch system

#PBS -q !TBG_queue
#PBS -l walltime=!TBG_wallTime
# Sets batch job's name
#PBS -N !TBG_jobName
#PBS -l nodes=!TBG_nodes:ppn=!TBG_coresPerNode
# send me a mail on (b)egin, (e)nd, (a)bortion
#PBS -m !TBG_mailSettings -M !TBG_mailAddress
#PBS -d !TBG_dstPath
#PBS -n

#PBS -o stdout
#PBS -e stderr

## calculation are done by tbg ##
.TBG_queue="k80"
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_mailSettings=${MY_MAILNOTIFY:-"n"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}

# 8 gpus per node if we need more than 8 gpus else same count as TBG_tasks
.TBG_gpusPerNode=`if [ $TBG_tasks -gt 8 ] ; then echo 8; else echo $TBG_tasks; fi`

#number of cores per parallel node / default is 2 cores per gpu on k80 queue
.TBG_coresPerNode="$(( TBG_gpusPerNode * 2 ))"

# use ceil to caculate nodes
.TBG_nodes="$(( ( TBG_tasks + TBG_gpusPerNode -1 ) / TBG_gpusPerNode))"
## end calculations ##

# overwrite .profile
.TBG_profileFile=$TBG_profileFile
profileFile=!TBG_profileFile
profileFile=${profileFile:-"$HOME/picongpu.profile"}

set -o pipefail
echo 'Running program...'

cd !TBG_dstPath

export MODULES_NO_OUTPUT=1
source $profileFile
if [ $? -ne 0 ] ; then
  echo "Error: $profileFile not found!"
  exit 1
fi
unset MODULES_NO_OUTPUT

#set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput

#wait that all nodes see ouput folder
sleep 1

if [ $? -eq 0 ] ; then
  mpiexec --prefix $MPIHOME -x LIBRARY_PATH -x LD_LIBRARY_PATH -tag-output --display-map -am !TBG_dstPath/tbg/openib.conf --mca mpi_leave_pinned 0 -npernode !TBG_gpusPerNode -n !TBG_tasks nvprof --profile-from-start off -o profile.%q{OMPI_COMM_WORLD_RANK}.nvprof !TBG_dstPath/bin/!TBG_program !TBG_author !TBG_programParams
fi

#mpiexec --prefix $MPIHOME -x LIBRARY_PATH -x LD_LIBRARY_PATH -npernode !TBG_gpusPerNode -n !TBG_tasks killall -9 !TBG_program

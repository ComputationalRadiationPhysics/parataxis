#!/usr/bin/env bash
# Copyright 2013-2016 Axel Huebl, Richard Pausch, Alexander Grund
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
# along with ParaTAXIS.
# If not, see <http://www.gnu.org/licenses/>.
#

# PIConGPU batch script for taurus' SLURM batch system

#SBATCH --partition=!TBG_queue
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName
#SBATCH --nodes=!TBG_nodes
#SBATCH --ntasks=!TBG_tasks
#SBATCH --mincpus=!TBG_mpiTasksPerNode
#SBATCH --cpus-per-task=!TBG_coresPerGPU
#SBATCH --mem-per-cpu=2583
#SBATCH --gres=gpu:!TBG_gpusPerNode
# send me mails on BEGIN, END, FAIL, REQUEUE, ALL,
# TIME_LIMIT, TIME_LIMIT_90, TIME_LIMIT_80 and/or TIME_LIMIT_50
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --workdir=!TBG_dstPath

#SBATCH -o stdout
#SBATCH -e stderr

## calculations will be performed by tbg ##
.TBG_queue="gpu2"

# settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"ALL"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}

# 4 gpus per node
.TBG_gpusPerNode=`if [ $TBG_tasks -gt 4 ] ; then echo 4; else echo $TBG_tasks; fi`

# number of cores to block per GPU - we got 6 cpus per gpu
#   and we will be accounted 6 CPUs per GPU anyway
.TBG_coresPerGPU=6

# We only start 1 MPI task per GPU
.TBG_mpiTasksPerNode="$(( TBG_gpusPerNode * 1 ))"

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_gpusPerNode -1 ) / TBG_gpusPerNode))"

## end calculations ##

# overwrite .profile
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

echo "`hostname`" > hostname.txt

# we are not sure if the current bullxmpi/1.2.4.3 catches pinned memory correctly
#   support ticket [Ticket:2014052241001186] srun: mpi mca flags
#   see bug https://github.com/ComputationalRadiationPhysics/picongpu/pull/438
export OMPI_MCA_mpi_leave_pinned=0

# Run CUDA memtest to check GPU's health
#srun -K1 !TBG_dstPath/picongpu/bin/cuda_memtest.sh

if [ $? -eq 0 ] ; then
  srun -K1 !TBG_dstPath/bin/!TBG_program !TBG_author !TBG_programParams
fi


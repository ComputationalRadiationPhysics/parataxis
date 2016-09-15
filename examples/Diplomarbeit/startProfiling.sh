#!/usr/bin/env bash

rm -rf ~/xrtTestsProfiling
export TBG_TPLFILE=submit/hypnos-hzdr/k20_nvprof_profile.tpl
export TBG_CFGFILE=submit/4gpuProfiling.cfg
export TBG_SUBMIT=qsub

buildSystem/runTests.py -e examples/Diplomarbeit/ -o ~/xrtTestsProfiling -t "DoubleSlitRandPos" -t "DoubleSlitRandPosSingle" -j -DCUDA_ARCH=sm_35 -DXRT_NVPROF_NUM_TS=5 -DXRT_NVPROF_START_TS=2998 -r --compile-only && \
cd ~/xrtTestsProfiling/installed/DiplomarbeitSim_cmake1 && \
tbg -t -s -c ~/xrtTestsProfiling/output/profilingDouble && \
cd ~/xrtTestsProfiling/installed/DiplomarbeitSim_cmake2 && \
tbg -t -s -c ~/xrtTestsProfiling/output/profilingSingle && \
echo "All submitted"


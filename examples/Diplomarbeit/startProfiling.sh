#!/usr/bin/env bash

rm -rf ~/xrtTestsProfiling
export TBG_TPLFILE=submit/hypnos-hzdr/k20_nvprof_profile.tpl
export TBG_CFGFILE=submit/8gpuProfiling.cfg
export TBG_SUBMIT=qsub
outputFolder = "$HOME/xrtTestsProfiling"

buildSystem/runTests.py -e examples/Diplomarbeit/ -o "$outputFolder" -t "DoubleSlitRandPos" -t "DoubleSlitRandPosSingle" -j -DCUDA_ARCH=sm_35 -DXRT_NVPROF_NUM_TS=5 -DXRT_NVPROF_START_TS=2998 -r --compile-only && \
cd "$outputFolder"/installed/DiplomarbeitSim_cmake1 && \
tbg -t -s -c "$outputFolder"/output/profilingDouble && \
cd "$outputFolder"/installed/DiplomarbeitSim_cmake2 && \
tbg -t -s -c "$outputFolder"/output/profilingSingle && \
echo "All submitted"


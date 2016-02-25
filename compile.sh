#!/usr/bin/env bash

this_dir="$( cd "$(dirname "$0")" ; pwd )"

exeDescription="Compile given example(s)"

. $this_dir/buildSystem/commonCmds.sh
. $this_dir/buildSystem/color.sh
. $this_dir/buildSystem/compileOptions.sh
. $this_dir/buildSystem/runParallel.sh

parseCompileOptions "$@"

descriptions=()
cmds=()
outputFiles=()

# example compile loop #########################################################
i=0
for exampleFolder in $list_examples; do
    example_name=$(getNameFromFolder "$srcDir/$exampleFolder")

    testFlag_cnt=0
    if [ -f "$srcDir/$exampleFolder/cmakeFlags" ]; then
        testFlag_cnt=`$srcDir/$exampleFolder/cmakeFlags -l`
    fi

    testFlagNr=0
    while [ $testFlagNr -lt $testFlag_cnt ] ; do
        buildDir="$destinationDir/build_"$example_name"_cmakePreset_$testFlagNr"
        mkdir -p $buildDir
        
        descriptions+=("$example_name ${testFlagNr}")
        cmds+=("\"$this_dir/compileExampleSet.sh\" -f -t $testFlagNr \"$srcDir/$exampleFolder\" \"$buildDir\"")
        outputFiles+=("$buildDir/compile.log")

        testFlagNr=$(( testFlagNr + 1 ))
    done
done

runParallel "$numParallel" "$descriptions" "$cmds" "$outputFiles"

# output errors
myError=0
errorTxt=""

for bT in `ls -w1 -d "$destinationDir/"build_*`; do
    returnCodeFile="$bT/returnCode"
    if [ ! -f "$returnCodeFile" ]; then
        thisError=99
    else
        thisError=`cat "$returnCodeFile"`
    fi
    if [ $thisError -ne 0 ]; then
        errorTxt="$errorTxt\n$bT"
    fi

    myError=$(( myError + thisError ))
done

if [ $myError -ne 0 ]; then
    printError "Errors occured!\nFolders:$errorTxt"
    thumbs_down
else
    echo_g "Everything fine!"
    thumbs_up
fi

exit $myError


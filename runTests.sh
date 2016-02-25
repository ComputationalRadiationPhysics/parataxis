#!/usr/bin/env bash

this_dir="$( cd "$(dirname "$0")" ; pwd )"

exeDescription="Compile given example(s) and run its test(s)"

. $this_dir/buildSystem/commonCmds.sh
. $this_dir/buildSystem/color.sh
. $this_dir/buildSystem/testOptions.sh

parseTestOptions "$@"

errors=0
for exampleFolder in $list_examples; do

    examplePath="$srcDir/$exampleFolder"
    exampleName=$(getNameFromFolder "$examplePath")
    
    if [ ! -f "$examplePath/documentation.yml" ]; then
        echo_r "No documentation found for '$exampleName'. Skipping..."
        continue
    fi
    
    set -o pipefail
    echo $this_dir/buildSystem/execTests.py -e "$examplePath" -o "$destinationDir" "$testOptions"
    $this_dir/buildSystem/execTests.py -e "$examplePath" -o "$destinationDir" $testOptions \
        2>&1 | tee "$destinationDir/${exampleName}_testOut.log"
    
    if [ $? -ne 0 ]; then
        errors=$(( errors + 1 ))
    fi

done

if [ $errors -ne 0 ]; then
    printError "Errors occured!"
    thumbs_down
else
    echo_g "Everything fine!"
    thumbs_up
fi

exit $errors


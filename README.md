# xrt
X-Ray Tracing with libPMacc

## Compile suite

There is an automated compile suite that can be used to compile one or multiple examples.
Compiling all examples can help to verify that a change does not break existing code.
Please refer to the help text of each individual script (`-h` or `--help`) for detailed explanations of the options

- `compile.sh`: Main script for compilation   
   Usage: `./compile.sh [OPTION] src_dir dest_dir`   
   For a full test of the compilation of the code use for example: `./compile.sh -l -j 4 ./examples $HOME/buildTest`
   
- `runTests.sh`: Main script for running test cases   
   Usage: `./runTests.sh [OPTION] -e src_dir -o dest_dir -- [TEST_OPTIONS]`   
   For running all tests of all examples use for example: `./runTests.sh -l -e ./examples -o $HOME/buildTest -- -s qsub -t submit/hypnos-hzdr/k80_profile.tpl`   
   Note: If you specify the same output directory as for `compile.sh` it will use the compiled programs if possible
   
- `configureExample.sh`: Configures an example into a build directory   
   Usage: `./configureExample.sh src_dir dest_dir`   
   You can use `-t <num>` to use pre-defined cmake flags and/or `-c <flag>` to define additional ones
   
- `compileExampleSet.sh`: Executes `configureExample.sh` and also calls `make install` from withing the build directory   
   Usage: Same as `configureExample.sh`
   
- `buildSystem/execTests.py`: Python 3 script that executes tests for 1 example   
   Used by `runTests.sh` but can be manually invoked.
   

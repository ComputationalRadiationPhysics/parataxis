# ParaTAXIS
**PARA**llel **T**racer for **A**rbitrary **X**-ray **I**nteraction and **S**cattering

This is a simulation framework for X-Ray Tracing with libPMacc. The code structure is almost identical to PIConGPU (https://github.com/ComputationalRadiationPhysics/picongpu) so most of the wiki entries apply here too.

The code uses C++11 and CUDA, so you need conforming compilers.

##Quickstart:

The code can be configured and compiled as it is and will run a basic example. But you mainly want to configure it yourself. This can be done by creating or modifying `*.param` files. Those are located in `include/simulation_defines/param`. You could modify those directly but it is better to use the override mechanism provided by the project. This allows you to create a folder where you place the param-files that will be used instead of the original ones:

- Use one of the examples in the `examples` folder as a starting point or create a new folder to start from scratch. Create a folder `include/simulation_defines/param` inside it, and copy the param-files you want to modify from the main repository.
- Adjust the parameters to your requirements. Everything must be valid C++ but often you only need to change numbers or assignments as documented inside the files.
- Call `./compileExampleSet.sh <src_dir> <dest_dir>` to configure, compile and install your configuration into `<dest_dir>`. This will also copy all the unchanged `simulation_defines` files into that folder for later reference
- Now change into that folder (`cd <dest_dir>`) and execute PIConGPUs `tbg` tool (best to have it in your `PATH`) like `tbg -s bash -c submit/0001gpus.cfg -t submit/bash/bash_mpirun.tpl <runOutputDir>` to run it. You may also want to adjust the `cfg`-file to use the correct runtime parameters. Note: You can copy on of the `cfg.in` from on of the examples into your example folder and adjust it just like the param-files.

## Compile suite

There is an automated compile suite that can be used to compile one or multiple examples.
Compiling all examples can help to verify that a change does not break existing code.
Please refer to the help text of each individual script (`-h` or `--help`) for detailed explanations of the options
   
- `buildSystem/runTests.py`: Main script for running test cases   
   Usage: `buildSystem/runTests.py -e <src_dir> -o <dest_dir> [other options]`   
   For running all tests of all examples use for example: `TBG_SUBMIT=qsub TBG_TPLFILE=submit/hypnos-hzdr/k20_profile.tpl buildSystem/runTests.py -e ./examples -o $HOME/buildTest --all`   
   Note: You can limit this to to compile-tests only (no runtime tests) with `--compile-only`. The script requires Python3 and YAML support. Some of the test-cases additionally need HDF5 and NumPy libraries.
   
- `configureExample.sh`: Configures an example into a build directory   
   Usage: `./configureExample.sh <src_dir> <dest_dir>`   
   You can use `-t <num>` to use pre-defined cmake flags and/or `-c <flag>` to define additional ones
   
- `compileExampleSet.sh`: Executes `configureExample.sh` and also calls `make install` from withing the build directory   
   Usage: Same as `configureExample.sh`
   

This build system uses YAML files for describing tests. The name must be `documentation.yml` in the root of an examples directory.
There also must be an executable file called `cmakeFlags` in the same folder which returns
the number of presets with `-l`, all presets with `-ll` and a given preset by its index `<index>` depending on its first command line param.    
The global `buildSystem` folder and optionally a `testData` folder from inside the examples root dir is copied to the install path during install phase.

**Structure of YAML file with explanation**

- `#?` means an optional entry
- Names must be unique


    example:         # Meta data start
      name:          # 
      short:         #? Short name (folder name), derived from name if not given
      author:        #
      description:   #?

    compile:         # Compile tests
      - cmakeFlags:  # List of CMake settings (array indices into `cmakeFlags` file)
                     # Ranges like `2-4` are allowed and treated as if `[2, 3, 4]`

    tests:           # List of Run-Time Tests
      - name:        # 
        description: #?
        cmakeFlag:   # CMake setting index used
        cfgFile:     # Cfg file to use 
        pre-run:     #? List of shell commands to run inside the install folder before execution. List of lists get flattened
        post-run:    #? List of shell commands to run inside the output folder after execution. List of lists get flattened
        dependency:  #? Name of the Run-Time tests this one depends on (only 1 level og dependencies allowed)
        
**Predefined environment variables during compilation**

- `TEST_BASE_BUILD_PATH`: Folder with all build folders
- `TEST_BUILD_PATH`     : Current build folder
- `TEST_INSTALL_PATH`   : Install target folder
- `TEST_CMAKE_FLAGS`    : String of CMAKE flags used from `cmakeFlags` file

**Additional environment variables during runtime tests**

- `TEST_NAME`           : Name of the runtime test from YAML
- `TEST_OUTPUT_PATH`    : Path in which the test is run
- `TEST_SIMOUTPUT_PATH` : Path in which simulation data is put
- `TEST_GRID_SIZE`      : Size of the grid from CFG file


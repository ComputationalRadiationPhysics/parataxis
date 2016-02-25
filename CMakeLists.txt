cmake_minimum_required(VERSION 2.8.12.2)

################################################################################
# Project 
################################################################################
project (XRayTracing)

set(PICONGPU_ROOT ~/picongpu CACHE PATH "Path to picongpu base directory")
#set helper pathes to find libraries and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{MPI_ROOT}" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")
# own modules for find_packages
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PICONGPU_ROOT}/thirdParty/cmake-modules/)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 

################################################################################
# Find CUDA 
################################################################################

find_package(CUDA 5.0 REQUIRED)

set(CUDA_ARCH sm_20 CACHE STRING "set GPU architecture" )
set(CUDA_FTZ "--ftz=false" CACHE STRING "Set flush to zero for GPU")
set(CUDA_MATH --use_fast_math CACHE STRING "use intrinsic GPU math functions" )
option(CUDA_SHOW_REGISTER "show kernel register and create PTX" OFF)
option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps. (folder: nvcc_tmp)" OFF)
option(CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)

if(CUDA_SHOW_CODELINES)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --source-in-ptx -Xcompiler -rdynamic -lineinfo)
    set(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
endif(CUDA_SHOW_CODELINES)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${nvcc_flags} -arch=${CUDA_ARCH} ${CUDA_MATH} ${CUDA_FTZ})
if(CUDA_SHOW_REGISTER)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xptxas=-v)
endif(CUDA_SHOW_REGISTER)

if(CUDA_KEEP_FILES)
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/nvcc_tmp")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "${PROJECT_BINARY_DIR}/nvcc_tmp")
endif(CUDA_KEEP_FILES)

if(NOT CUDA_VERSION VERSION_LESS "7.5")
    set(CUDA_RELAXED_CONSTEXPR "--expt-relaxed-constexpr")
elseif(NOT CUDA_VERSION VERSION_LESS "7.0")
    set(CUDA_RELAXED_CONSTEXPR "--relaxed-constexpr")
else()
    set(CUDA_RELAXED_CONSTEXPR "")
endif()


# Does not work with -G (bug in nvcc with std::map, -G and --relaxed-constexpr: #1690424)
if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-G")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_RELAXED_CONSTEXPR}")
else()
    message(WARNING "-G flag found for CUDA. Cannot use --relaxed-constexpr. Expect compiletime warnings!")
endif()

################################################################################
# VampirTrace
################################################################################

option(VAMPIR_ENABLE "Create with VampirTrace support" OFF)

# set filters: please do NOT use line breaks WITHIN the string!
set(VT_INST_FILE_FILTER
    "stl,usr/include,libgpugrid,vector_types.h,Vector.hpp,DeviceBuffer.hpp,DeviceBufferIntern.hpp,Buffer.hpp,StrideMapping.hpp,StrideMappingMethods.hpp,MappingDescription.hpp,AreaMapping.hpp,AreaMappingMethods.hpp,ExchangeMapping.hpp,ExchangeMappingMethods.hpp,DataSpace.hpp,Manager.hpp,Manager.tpp,Transaction.hpp,Transaction.tpp,TransactionManager.hpp,TransactionManager.tpp,Vector.tpp,Mask.hpp,ITask.hpp,EventTask.hpp,EventTask.tpp,StandardAccessor.hpp,StandardNavigator.hpp,HostBuffer.hpp,HostBufferIntern.hpp"
    CACHE STRING "VampirTrace: Files to exclude from instrumentation")
set(VT_INST_FUNC_FILTER
    "vector,Vector,dim3,GPUGrid,execute,allocator,Task,Manager,Transaction,Mask,operator,DataSpace,PitchedBox,Event,new,getGridDim,GetCurrentDataSpaces,MappingDescription,getOffset,getParticlesBuffer,getDataSpace,getInstance"
    CACHE STRING "VampirTrace: Functions to exclude from instrumentation")

if(VAMPIR_ENABLE)
    message(STATUS "Building with VampirTrace support")
    set(VAMPIR_ROOT "$ENV{VT_ROOT}")
    if(NOT VAMPIR_ROOT)
        message(FATAL_ERROR "Environment variable VT_ROOT not set!")
    endif(NOT VAMPIR_ROOT)

    # compile flags
    execute_process(COMMAND $ENV{VT_ROOT}/bin/vtc++ -vt:hyb -vt:showme-compile
                    OUTPUT_VARIABLE VT_COMPILEFLAGS
                    RESULT_VARIABLE VT_CONFIG_RETURN
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT VT_CONFIG_RETURN EQUAL 0)
        message(FATAL_ERROR "Can NOT execute 'vtc++' at $ENV{VT_ROOT}/bin/vtc++ - check file permissions")
    endif()
    # link flags
    execute_process(COMMAND $ENV{VT_ROOT}/bin/vtc++ -vt:hyb -vt:showme-link
                    OUTPUT_VARIABLE VT_LINKFLAGS
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    # bugfix showme
    string(REPLACE "--as-needed" "--no-as-needed" VT_LINKFLAGS "${VT_LINKFLAGS}")

    # modify our flags
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${VT_LINKFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VT_COMPILEFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions-exclude-file-list=${VT_INST_FILE_FILTER}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions-exclude-function-list=${VT_INST_FUNC_FILTER}")

    # nvcc flags (rly necessary?)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -Xcompiler=-finstrument-functions,-finstrument-functions-exclude-file-list=\\\"${VT_INST_FILE_FILTER}\\\"
        -Xcompiler=-finstrument-functions-exclude-function-list=\\\"${VT_INST_FUNC_FILTER}\\\"
        -Xcompiler=-DVTRACE -Xcompiler=-I\\\"${VT_ROOT}/include/vampirtrace\\\"
        -v)

    # for manual instrumentation and hints that vampir is enabled in our code
    add_definitions(-DVTRACE)

    # titan work around: currently (5.14.4) the -D defines are not provided by -vt:showme-compile
    add_definitions(-DMPICH_IGNORE_CXX_SEEK)
endif(VAMPIR_ENABLE)

################################################################################
# Score-P
################################################################################

option(SCOREP_ENABLE "Create with Score-P support" OFF)

if(SCOREP_ENABLE)
    message(STATUS "Building with Score-P support")
    set(SCOREP_ROOT "$ENV{SCOREP_ROOT}")
    if(NOT SCOREP_ROOT)
        message(FATAL_ERROR "Environment variable SCOREP_ROOT not set!")
    endif(NOT SCOREP_ROOT)

    # compile flags
    execute_process(COMMAND $ENV{SCOREP_ROOT}/bin/scorep-config --nocompiler --cflags
                    OUTPUT_VARIABLE SCOREP_COMPILEFLAGS
                    RESULT_VARIABLE SCOREP_CONFIG_RETURN
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT SCOREP_CONFIG_RETURN EQUAL 0)
        message(FATAL_ERROR "Can NOT execute 'scorep-config' at $ENV{SCOREP_ROOT}/bin/scorep-config - check file permissions")
    endif()

    # link flags
    execute_process(COMMAND $ENV{SCOREP_ROOT}/bin/scorep-config --cuda --mpp=mpi --ldflags
                    OUTPUT_VARIABLE SCOREP_LINKFLAGS
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    # libraries
    execute_process(COMMAND $ENV{SCOREP_ROOT}/bin/scorep-config --cuda --mpp=mpi --libs
                    OUTPUT_VARIABLE SCOREP_LIBFLAGS
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(STRIP "${SCOREP_LIBFLAGS}" SCOREP_LIBFLAGS)

    # subsystem iniialization file
    execute_process(COMMAND $ENV{SCOREP_ROOT}/bin/scorep-config --cuda --mpp=mpi --adapter-init
                    OUTPUT_VARIABLE SCOREP_INIT_FILE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    file(WRITE ${CMAKE_BINARY_DIR}/scorep_init.c "${SCOREP_INIT_FILE}")

    if(SCOREP_ENABLE)
        set(SCOREP_SRCFILES "${CMAKE_BINARY_DIR}/scorep_init.c")
    endif(SCOREP_ENABLE)

    # modify our flags
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${SCOREP_LINKFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SCOREP_COMPILEFLAGS}")
endif(SCOREP_ENABLE)

################################################################################
# Build type (debug, release)
################################################################################

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()
 
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    message(STATUS "Release version")

    add_definitions(-DNDEBUG)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcompiler=-pthread")
else()
    message(STATUS "Debug version")

    set(CMAKE_BUILD_TYPE Debug)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -g "-Xcompiler=-g,-pthread")
endif()

################################################################################
# Find MPI
################################################################################

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
list(APPEND LIBS ${MPI_CXX_LIBRARIES})

################################################################################
# Find Boost
################################################################################

find_package(Boost 1.49 REQUIRED COMPONENTS program_options regex system filesystem)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
list(APPEND LIBS ${Boost_LIBRARIES})
add_definitions(-DBOOST_RESULT_OF_USE_TR1)

################################################################################
# Find OpenMP
################################################################################

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

################################################################################
# Find mallocMC
################################################################################
find_package(mallocMC 2.1.0 QUIET)

if(NOT mallocMC_FOUND)
    message(STATUS "Using mallocMC from ${PICONGPU_ROOT}/thirdParty/ directory")
    set(MALLOCMC_ROOT "${PICONGPU_ROOT}/thirdParty/mallocMC" CACHE PATH "Path to MallocMC")
    find_package(mallocMC 2.1.0 REQUIRED)
endif(NOT mallocMC_FOUND)

include_directories(SYSTEM ${mallocMC_INCLUDE_DIRS})
add_definitions(${mallocMC_DEFINITIONS})
list(APPEND LIBS ${mallocMC_LIBRARIES})

################################################################################
# PMacc options
################################################################################

find_path(PMACC_ROOT_DIR
  NAMES include/types.h
  PATHS "${PICONGPU_ROOT}/src/libPMacc"
  DOC "libPMacc root location."
  )

include_directories(${PMACC_ROOT_DIR}/include)

option(PMACC_BLOCKING_KERNEL "Activate checks for every kernel call and synchronize after every kernel call" OFF)
if(PMACC_BLOCKING_KERNEL)
    add_definitions(-DPMACC_SYNC_KERNEL=1)
endif(PMACC_BLOCKING_KERNEL)

set(PMACC_VERBOSE "0" CACHE STRING "Set verbosity level for libPMacc")
add_definitions(-DPMACC_VERBOSE_LVL=${PMACC_VERBOSE})

################################################################################
# PARAM (overwrite) defines
################################################################################

if(PARAM_OVERWRITES)
    foreach(param ${PARAM_OVERWRITES})
       add_definitions(${param})
    endforeach(param)
endif(PARAM_OVERWRITES)

################################################################################
# XRT
################################################################################

set(XRT_VERBOSE "1" CACHE STRING "Set verbosity level for XRT")
add_definitions(-DXRT_VERBOSE_LVL=${XRT_VERBOSE})
option(XRT_DEBUG_RNG "Output test of RNG provider" OFF)
option(XRT_CHECK_PHOTON_CT "Checks if the number of photons generated can fit into the memory" ON)
set(XRT_NVPROF_START_TS 200 CACHE STRING "Start profiling at this timestep")
set(XRT_NVPROF_NUM_TS 5 CACHE STRING "Number of timesteps to profile. 0 = Disabled") 
if(XRT_DEBUG_RNG)
    add_definitions(-DXRT_DEBUG_RNG)
endif()
if(XRT_CHECK_PHOTON_CT)
    add_definitions(-DXRT_CHECK_PHOTON_CT)
endif()
add_definitions(-DXRT_NVPROF_START_TS=${XRT_NVPROF_START_TS} -DXRT_NVPROF_NUM_TS=${XRT_NVPROF_NUM_TS})

################################################################################
# Warnings
################################################################################

# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    # new warning in gcc 4.8 (flag ignored in previous version)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
# ICC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_FENV_H")
# PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
endif()

################################################################################
# PNGwriter
################################################################################

# find PNGwriter installation
find_package(PNGwriter 0.5.5)

if(PNGwriter_FOUND)
    include_directories(SYSTEM ${PNGwriter_INCLUDE_DIRS})
    list(APPEND PNGwriter_DEFINITIONS "-DXRT_ENABLE_PNG=1")
    add_definitions(${PNGwriter_DEFINITIONS})
    list(APPEND LIBS ${PNGwriter_LIBRARIES})
endif()

################################################################################
# TiffWriter
################################################################################

find_package(TiffWriter)

if(TiffWriter_FOUND)
    include_directories(${TiffWriter_INCLUDE_DIRS})
    list(APPEND TiffWriter_DEFINITIONS "-DXRT_ENABLE_TIFF=1")
    add_definitions(${TiffWriter_DEFINITIONS})
    list(APPEND LIBS ${TiffWriter_LIBRARIES})
endif()

################################################################################
# C++11 Support
################################################################################

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")


################################################################################
# Check for XRT_EXTENSION_PATH
################################################################################

find_path(XRT_EXTENSION_PATH
    NAMES include/simulation_defines/param/precision.param
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}"
    DOC "search PATH to extension folder"
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    )

include_directories(include)
if(XRT_EXTENSION_PATH)
    include_directories(BEFORE ${XRT_EXTENSION_PATH}/include)
    file(GLOB CFG_Files_Ext "${XRT_EXTENSION_PATH}/submit/*.cfg.in")
else()
    SET(CFG_Files_Ext)
endif()

################################################################################
# Compile & Link
################################################################################

file(GLOB CUDASRCFILES "*.cu")
file(GLOB SRCFILES "*.cpp")

cuda_add_executable(${PROJECT_NAME}
    ${CUDASRCFILES}
    ${SRCFILES}
    ${SCOREP_SRCFILES}
)

if(VAMPIR_ENABLE)
    set(LIBS vt-hyb ${LIBS} )
endif(VAMPIR_ENABLE)

target_link_libraries(${PROJECT_NAME} ${LIBS} m)

if(SCOREP_ENABLE)
    # Score-P libraries must be linked after the object to prevent symbols
    # from being stripped on Titan
    target_link_libraries(${PROJECT_NAME} ${SCOREP_LIBFLAGS})
endif(SCOREP_ENABLE)

################################################################################
# Install
################################################################################

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/submit"
        DESTINATION ${CMAKE_INSTALL_PREFIX}
        PATTERN *.in EXCLUDE)

file(GLOB CFG_Files "submit/*.cfg.in")
list(APPEND CFG_Files ${CFG_Files_Ext})
list(REMOVE_DUPLICATES CFG_Files)

foreach(file ${CFG_Files})
    get_filename_component(fileNoExt ${file} NAME_WE)
    configure_file(${file} "submit/${fileNoExt}.cfg" @ONLY)
endforeach()

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/submit/submitAction.sh"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/submit/"
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE)


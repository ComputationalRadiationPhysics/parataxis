cmake_minimum_required(VERSION 2.8.12.2)

OPTION(VAMPIR_ENABLE "create with vampir support" OFF)

IF(VAMPIR_ENABLE)
    message("[CONFIG]  build program with vampir support")
    set(CMAKE_CXX_COMPILER "vtc++")
    set(CMAKE_CXX_INST_FILE_FILTER "stl,usr/include,vector_types.h,Vector.hpp,DeviceBuffer.hpp,DeviceBufferIntern.hpp,Buffer.hpp,StrideMapping.hpp,StrideMappingMethods.hpp,MappingDescription.hpp,AreaMapping.hpp,AreaMappingMethods.hpp,ExchangeMapping.hpp,ExchangeMappingMethods.hpp,DataSpace.hpp,Manager.hpp,Manager.tpp,Transaction.hpp,Transaction.tpp,TransactionManager.hpp,TransactionManager.tpp,Vector.tpp,Mask.hpp,ITask.hpp,EventTask.hpp,EventTask.tpp,StandardAccessor.hpp,StandardNavigator.hpp,HostBuffer.hpp,HostBufferIntern.hpp")
    set(CMAKE_CXX_INST_FUNC_FILTER "vector,Vector,dim3,PMacc,execute,allocator,Task,Manager,Transaction,Mask,operator,DataSpace,PitchedBox,Event,new,getGridDim,GetCurrentDataSpaces,MappingDescription,getOffset,getParticlesBuffer,getDataSpace,getInstance")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -vt:hyb -L/$ENV{VT_ROOT}/lib -finstrument-functions-exclude-file-list=${CMAKE_CXX_INST_FILE_FILTER} -finstrument-functions-exclude-function-list=${CMAKE_CXX_INST_FUNC_FILTER} -DVTRACE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -vt:hyb -L/$ENV{VT_ROOT}/lib -finstrument-functions-exclude-file-list=${CMAKE_CXX_INST_FILE_FILTER} -finstrument-functions-exclude-function-list=${CMAKE_CXX_INST_FUNC_FILTER} -DVTRACE")
ENDIF(VAMPIR_ENABLE)

# Projekt name
project (XRayTracing)
SET(LIBS m)

SET(PICONGPU_ROOT ~/picongpu CACHE PATH "Path to picongpu base directory")
#set helper pathes to find libraries and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{MPI_ROOT}" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")
# own modules for find_packages
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PICONGPU_ROOT}/thirdParty/cmake-modules/)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 

INCLUDE_DIRECTORIES(include)

## MPI ##
FIND_PACKAGE(MPI REQUIRED)
INCLUDE_DIRECTORIES(${MPI_CXX_INCLUDE_PATH})
list(APPEND LIBS ${MPI_CXX_LIBRARIES})

## CUDA ##
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_ARCH sm_30 CACHE STRING "set GPU architecture" )
SET(CUDA_MATH --use_fast_math CACHE STRING "use intrinsic GPU math functions" )
OPTION(CUDA_SHOW_REGISTER "show kernel register and create PTX" OFF)
OPTION(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps. (folder: nvcc_tmp)" OFF)


SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${nvcc_flags} -arch=${CUDA_ARCH} ${CUDA_MATH})
IF(CUDA_SHOW_REGISTER)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xptxas=-v -Xopencc=-LIST:source=on)
ENDIF(CUDA_SHOW_REGISTER)

IF(CUDA_KEEP_FILES)
    FILE(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/nvcc_tmp")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "nvcc_tmp")
ENDIF(CUDA_KEEP_FILES)

## libPMacc ##
find_path(PMACC_ROOT_DIR
  NAMES include/types.h
  PATHS "${PICONGPU_ROOT}/src/libPMacc"
  DOC "libPMacc root location."
  )

INCLUDE_DIRECTORIES(${PMACC_ROOT_DIR}/include)

OPTION(PMACC_BLOCKING_KERNEL "activate checks for every kernel call and sync after every kernel call" OFF)
IF(PMACC_BLOCKING_KERNEL)
    ADD_DEFINITIONS(-DPMACC_SYNC_KERNEL=1)
ENDIF(PMACC_BLOCKING_KERNEL)

SET(PMACC_VERBOSE "0" CACHE STRING "set verbose level for libPMacc")
ADD_DEFINITIONS(-DPMACC_VERBOSE_LVL=${PMACC_VERBOSE})

## PNGwriter ##
find_package(PNGwriter REQUIRED)

include_directories(SYSTEM ${PNGwriter_INCLUDE_DIRS})
add_definitions(${PNGwriter_DEFINITIONS})
list(APPEND LIBS ${PNGwriter_LIBRARIES})

## Boost ##
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options regex system filesystem)
INCLUDE_DIRECTORIES(AFTER ${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
list(APPEND LIBS ${Boost_LIBRARIES})
ADD_DEFINITIONS(-DBOOST_RESULT_OF_USE_TR1)

## MallocMC ##
SET(MALLOCMC_ROOT "${PICONGPU_ROOT}/thirdParty/mallocMC" CACHE PATH "Path to MallocMC")
FIND_PACKAGE(mallocMC REQUIRED)
include_directories(SYSTEM ${mallocMC_INCLUDE_DIRS})
add_definitions(${mallocMC_DEFINITIONS})
list(APPEND LIBS ${mallocMC_LIBRARIES})

## XRT ##
set(XRT_VERBOSE "1" CACHE STRING "Set verbosity level for XRT")
add_definitions(-DXRT_VERBOSE_LVL=${XRT_VERBOSE})


################################################################################
# Warnings
################################################################################

# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
# ICCPROJECT_NAME 
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_FENV_H")
# PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
endif()

#################################################################################

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${INCLUDE_DIRECTORIES})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-std=c++11 -Wall)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -Xcompiler -Wall")
list(APPEND CUDA_NVCC_FLAGS "--relaxed-constexpr")

IF(VAMPIR_ENABLE)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler=-finstrument-functions,-finstrument-functions-exclude-file-list=stl,-finstrument-functions-exclude-function-list='GPUGrid,execute,allocator,Task,Manager,Transaction,Mask',-pthread )
ENDIF(VAMPIR_ENABLE)

file(GLOB CUDASRCFILES "*.cu")
file(GLOB SRCFILES "*.cpp")

cuda_add_executable(${PROJECT_NAME}
    ${CUDASRCFILES}
    ${SRCFILES}
)

IF(VAMPIR_ENABLE)
    SET(LIBS vt-hyb ${LIBS} )
ENDIF(VAMPIR_ENABLE)

target_link_libraries(${PROJECT_NAME} ${LIBS})

###install section###
INSTALL(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/submit"
        DESTINATION ${CMAKE_INSTALL_PREFIX}
        PATTERN *.in EXCLUDE)

file(GLOB CFG_Files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "submit/*.cfg.in")
FOREACH(file ${CFG_Files})
    get_filename_component(fileNoExt ${file} NAME_WE)
    message(STATUS ${fileNoExt})
    configure_file(${file} "submit/${fileNoExt}.cfg")
ENDFOREACH()

INSTALL(FILES "${CMAKE_CURRENT_SOURCE_DIR}/submit/submitAction.sh"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/submit/"
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE)


cmake_minimum_required(VERSION 2.8.12.2)

option(VAMPIR_ENABLE "create with vampir support" OFF)

if(VAMPIR_ENABLE)
    message("[CONFIG]  build program with vampir support")
    set(CMAKE_CXX_COMPILER "vtc++")
    set(CMAKE_CXX_INST_FILE_FILTER "stl,usr/include,vector_types.h,Vector.hpp,DeviceBuffer.hpp,DeviceBufferIntern.hpp,Buffer.hpp,StrideMapping.hpp,StrideMappingMethods.hpp,MappingDescription.hpp,AreaMapping.hpp,AreaMappingMethods.hpp,ExchangeMapping.hpp,ExchangeMappingMethods.hpp,DataSpace.hpp,Manager.hpp,Manager.tpp,Transaction.hpp,Transaction.tpp,TransactionManager.hpp,TransactionManager.tpp,Vector.tpp,Mask.hpp,ITask.hpp,EventTask.hpp,EventTask.tpp,StandardAccessor.hpp,StandardNavigator.hpp,HostBuffer.hpp,HostBufferIntern.hpp")
    set(CMAKE_CXX_INST_FUNC_FILTER "vector,Vector,dim3,PMacc,execute,allocator,Task,Manager,Transaction,Mask,operator,DataSpace,PitchedBox,Event,new,getGridDim,GetCurrentDataSpaces,MappingDescription,getOffset,getParticlesBuffer,getDataSpace,getInstance")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -vt:hyb -L/$ENV{VT_ROOT}/lib -finstrument-functions-exclude-file-list=${CMAKE_CXX_INST_FILE_FILTER} -finstrument-functions-exclude-function-list=${CMAKE_CXX_INST_FUNC_FILTER} -DVTRACE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -vt:hyb -L/$ENV{VT_ROOT}/lib -finstrument-functions-exclude-file-list=${CMAKE_CXX_INST_FILE_FILTER} -finstrument-functions-exclude-function-list=${CMAKE_CXX_INST_FUNC_FILTER} -DVTRACE")
endif(VAMPIR_ENABLE)

# Projekt name
project (XRayTracing)
set(LIBS m)

set(PICONGPU_ROOT ~/picongpu CACHE PATH "Path to picongpu base directory")
#set helper pathes to find libraries and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{MPI_ROOT}" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")
# own modules for find_packages
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PICONGPU_ROOT}/thirdParty/cmake-modules/)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 

include_directories(include)

## MPI ##
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
list(APPEND LIBS ${MPI_CXX_LIBRARIES})

## CUDA ##
find_package(CUDA REQUIRED)
set(CUDA_ARCH sm_20 CACHE STRING "set GPU architecture" )
set(CUDA_MATH --use_fast_math CACHE STRING "use intrinsic GPU math functions" )
option(CUDA_SHOW_REGISTER "show kernel register and create PTX" OFF)
option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps. (folder: nvcc_tmp)" OFF)


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${nvcc_flags} -arch=${CUDA_ARCH} ${CUDA_MATH})
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

## libPMacc ##
find_path(PMACC_ROOT_DIR
  NAMES include/types.h
  PATHS "${PICONGPU_ROOT}/src/libPMacc"
  DOC "libPMacc root location."
  )

include_directories(${PMACC_ROOT_DIR}/include)

option(PMACC_BLOCKING_KERNEL "activate checks for every kernel call and sync after every kernel call" OFF)
if(PMACC_BLOCKING_KERNEL)
    add_definitions(-DPMACC_SYNC_KERNEL=1)
endif(PMACC_BLOCKING_KERNEL)

set(PMACC_VERBOSE "0" CACHE STRING "set verbose level for libPMacc")
add_definitions(-DPMACC_VERBOSE_LVL=${PMACC_VERBOSE})

## PNGwriter ##
find_package(PNGwriter)

if(PNGwriter_FOUND)
    include_directories(SYSTEM ${PNGwriter_INCLUDE_DIRS})
    add_definitions(${PNGwriter_DEFINITIONS} -DXRT_ENABLE_PNG)
    list(APPEND LIBS ${PNGwriter_LIBRARIES})
endif()

## PNGwriter ##
find_package(TiffWriter)

if(TiffWriter_FOUND)
    include_directories(${TiffWriter_INCLUDE_DIRS})
    add_definitions(${TiffWriter_DEFINITIONS} -DXRT_ENABLE_TIFF)
    list(APPEND LIBS ${TiffWriter_LIBRARIES})
endif()

## Boost ##
find_package(Boost REQUIRED COMPONENTS program_options regex system filesystem)
include_directories(AFTER ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIR})
list(APPEND LIBS ${Boost_LIBRARIES})
add_definitions(-DBOOST_RESULT_OF_USE_TR1)

## MallocMC ##
set(MALLOCMC_ROOT "${PICONGPU_ROOT}/thirdParty/mallocMC" CACHE PATH "Path to MallocMC")
find_package(mallocMC REQUIRED)
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

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-std=c++11)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "${CUDA_RELAXED_CONSTEXPR}")

if(VAMPIR_ENABLE)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler=-finstrument-functions,-finstrument-functions-exclude-file-list=stl,-finstrument-functions-exclude-function-list='GPUGrid,execute,allocator,Task,Manager,Transaction,Mask',-pthread )
endif(VAMPIR_ENABLE)

file(GLOB CUDASRCFILES "*.cu")
file(GLOB SRCFILES "*.cpp")

cuda_add_executable(${PROJECT_NAME}
    ${CUDASRCFILES}
    ${SRCFILES}
)

if(VAMPIR_ENABLE)
    set(LIBS vt-hyb ${LIBS} )
endif(VAMPIR_ENABLE)

target_link_libraries(${PROJECT_NAME} ${LIBS})

###install section###
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/submit"
        DESTINATION ${CMAKE_INSTALL_PREFIX}
        PATTERN *.in EXCLUDE)

file(GLOB CFG_Files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "submit/*.cfg.in")
foreach(file ${CFG_Files})
    get_filename_component(fileNoExt ${file} NAME_WE)
    configure_file(${file} "submit/${fileNoExt}.cfg" @ONLY)
endforeach()

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/submit/submitAction.sh"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/submit/"
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE)


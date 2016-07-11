# - Find TiffWriter library,
#     a C++ library for creating Tiff images
#     https://github.com/ComputationalRadiationPhysics/halt
#
# Use this module by invoking find_package with the form:
#   find_package(TiffWriter
#     [version] [EXACT]     # Minimum or EXACT version, e.g. 0.1.0
#     [REQUIRED]            # Fail with an error if TiffWriter or a required
#                           #   component is not found
#     [QUIET]               # Do not warn if this module was not found
#     [COMPONENTS <...>]    # Compiled in components: not implemented
#   )
#
# To provide a hint to this module where to find the TiffWriter installation,
# set the TIFFWRITER_ROOT environment variable.
#
# This module requires a valid installation of libTIFF
#
#
# This module will define the following variables:
#   TiffWriter_INCLUDE_DIRS    - Include directories for the TiffWriter headers.
#   TiffWriter_LIBRARIES       - TiffWriter libraries.
#   TiffWriter_FOUND           - TRUE if FindTiffWriterr found a working install
#   TiffWriter_VERSION         - Version in format Major.Minor.Patch
#   TiffWriter_DEFINITIONS     - Compiler definitions you should add with
#                               add_definitions(${TiffWriter_DEFINITIONS})
#

cmake_minimum_required(VERSION 2.8.5)

find_path(TiffWriter_ROOT_DIR
    NAMES tiffWriter/tiffWriter.hpp
    PATHS ${TIFFWRITER_ROOT} ENV TIFFWRITER_ROOT
    PATH_SUFFIXES src include src/include
    DOC "TiffWriter ROOT directory")

# find libTIFF install #########################################################
find_package(TIFF)

if(NOT TIFF_FOUND)
    message(WARNING "Did not find libTIFF. Cannot use TiffWriter")
elseif(TiffWriter_ROOT_DIR)
    set(TiffWriter_INCLUDE_DIRS ${TiffWriter_ROOT_DIR} ${TIFF_INCLUDE_DIR})
    set(TiffWriter_LIBRARIES ${TIFF_LIBRARIES})

    set(TiffWriter_DEFINITIONS )

    # version
    file(STRINGS "${TiffWriter_ROOT_DIR}/tiffWriter/tiffWriter.hpp" TiffWriter_VERSION_MAJOR_HPP REGEX "#define TIFFWRITER_VERSION_MAJOR ")
    file(STRINGS "${TiffWriter_ROOT_DIR}/tiffWriter/tiffWriter.hpp" TiffWriter_VERSION_MINOR_HPP REGEX "#define TIFFWRITER_VERSION_MINOR ")
    file(STRINGS "${TiffWriter_ROOT_DIR}/tiffWriter/tiffWriter.hpp" TiffWriter_VERSION_PATCH_HPP REGEX "#define TIFFWRITER_VERSION_PATCH ")
    if(TiffWriter_VERSION_MAJOR_HPP)
        string(REGEX MATCH "([0-9]+)" TiffWriter_VERSION_MAJOR ${TiffWriter_VERSION_MAJOR_HPP})
        string(REGEX MATCH "([0-9]+)" TiffWriter_VERSION_MINOR ${TiffWriter_VERSION_MINOR_HPP})
        string(REGEX MATCH "([0-9]+)" TiffWriter_VERSION_PATCH ${TiffWriter_VERSION_PATCH_HPP})

        set(TiffWriter_VERSION "${TiffWriter_VERSION_MAJOR}.${TiffWriter_VERSION_MINOR}.${TiffWriter_VERSION_PATCH}")
    else()
        set(TiffWriter_VERSION "0.0.1")
    endif()

    unset(TiffWriter_VERSION_MAJOR_HPP)
    unset(TiffWriter_VERSION_MINOR_HPP)
    unset(TiffWriter_VERSION_PATCH_HPP)

else()
    message(STATUS "Can NOT find TiffWriter - set TIFFWRITER_ROOT")
endif()

###############################################################################
# FindPackage Options
###############################################################################

# handles the REQUIRED, QUIET and version-related arguments for find_package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TiffWriter
    FOUND_VAR TiffWriter_FOUND
    REQUIRED_VARS TiffWriter_LIBRARIES TiffWriter_INCLUDE_DIRS
    VERSION_VAR TiffWriter_VERSION
)

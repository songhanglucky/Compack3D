
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Enable CUDA device acceleration")
#cmake_policy(SET CMP0074 NEW)
#cmake_policy(SET CMP0079 NEW)

project(Compac3D
    DESCRIPTION "Parallel linear solver for compact numerical methods"
)


########
### Compile options
########
set(COMPACK3D_ENABLE_DEVICE_CUDA             OFF CACHE BOOL   "Enable CUDA")
set(COMPACK3D_ENABLE_DEVICE_HIP              OFF CACHE BOOL   "Enable HIP")
set(COMPACK3D_ENABLE_DEVICE_AWARE_COMM       OFF CACHE BOOL   "Enable GPU-aware communication")
set(COMPACK3D_DEVICE_ARCH              "Default" CACHE STRING "Device architecture")
set(COMPACK3D_BUILD_UNIT_TESTS               OFF CACHE BOOL   "Compile unit tests")
set(COMPACK3D_BUILD_BENCHMARK_TESTS          OFF CACHE BOOL   "Compile benchmark tests")
set(COMPACK3D_BUILD_STENCIL_APIS             OFF CACHE BOOL   "Build APIs of Pade or explicit schemes")

if (${CMAKE_BUILD_TYPE} STREQUAL "Help")
    message(STATUS
        " Available configuration flags:\n"
        "     * COMPACK3D_ENABLE_DEVICE_CUDA        (BOOL)   Enable Nvidia or other CUDA-supported GPU acceleration (default: OFF)\n"
        "     * COMPACK3D_ENABLE_DEVICE_HIP         (BOOL)   Enable AMD or other HIP-supported GPU acceleration (default: OFF)\n"
        "     * COMPACK3D_ENABLE_DEVICE_AWARE_COMM  (BOOL)   Enable direct GPU-to-GPU communication (default: OFF)\n"
        "     * COMPACK3D_DEVICE_ARCH             (STRING)   Specify GPU architectures (default: \"Default\")\n"
        "     * COMPACK3D_BUILD_STENCIL_APIS        (BOOL)   Build APIs for Pade or explicit schemes\n"
        "     * COMPACK3D_BUILD_UNIT_TESTS          (BOOL)   Build unit tests\n"
        "     * COMPACK3D_BUILD_BENCHMARK_TESTS     (BOOL)   Build benchmark tests\n"
    )
endif()



########
### Configure device
########
set(tmp_var_device_count 0)
set(tmp_var_device_prog_model "None")
set(tmp_var_device_arch_label "n/a")
set(tmp_var_device_comm "Device-Host-to-Host-Device")
set(tmp_var_modules "Compack3D-core")
if (${COMPACK3D_BUILD_STENCIL_APIS})
    list(APPEND tmp_var_modules "Compack3D-stencil-APIs")
endif()
if (${COMPACK3D_BUILD_UNIT_TESTS})
    list(APPEND tmp_var_modules "Compack3D-unit-tests")
endif()
if (${COMPACK3D_BUILD_BENCHMARK_TESTS})
    list(APPEND tmp_var_modules "Compack3D-benchmark-tests")
endif()


if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    MATH(EXPR tmp_var_device_count "${tmp_var_device_count}+1")
    set(tmp_var_device_prog_model "CUDA")
endif()


if (${COMPACK3D_ENABLE_DEVICE_HIP})
    MATH(EXPR tmp_var_device_count "${tmp_var_device_count}+1")
    set(tmp_var_device_prog_model "HIP")
endif()


if (${tmp_var_device_count} GREATER 1)
    message(FATAL_ERROR
        " Compiling using more than one device programming model is not allowed.\n"
        "   COMPACK3D_ENABLE_DEVICE_CUDA = ${COMPACK3D_ENABLE_DEVICE_CUDA}\n"
        "   COMPACK3D_ENABLE_DEVICE_HIP  = ${COMPACK3D_ENABLE_DEVICE_HIP}\n"
    )
endif()


if    (${COMPACK3D_ENABLE_DEVICE_CUDA})                             #### CUDA
    set(tmp_var_device_prog_model "CUDA")
    add_compile_definitions(COMPACK3D_DEVICE_ENABLED_CUDA)
    if(${COMPACK3D_DEVICE_ARCH} STREQUAL "Default")          # Nvidia Tesla V100
        set(tmp_var_device_arch_label "Default (Nvidia Tesla V100)")
        add_compile_definitions(COMPACK3D_DEVICE_ARCH_V100)
        set(CMAKE_CUDA_ARCHITECTURES 70)
    elseif(${COMPACK3D_DEVICE_ARCH} STREQUAL "V100")         # Nvidia Tesla V100
        add_compile_definitions(COMPACK3D_DEVICE_ARCH_V100)
        set(tmp_var_device_arch_label "Nvidia Tesla V100")
        set(CMAKE_CUDA_ARCHITECTURES 70)
    else()
        message(FATAL_ERROR "Unsupported CUDA device architecture:\n"
            "     COMPACK3D_DEVICE_ARCH = ${COMPACK3D_DEVICE_ARCH}")
    endif()
elseif(${COMPACK3D_ENABLE_HIP})                               #### HIP
    add_compile_definitions(COMPACK3D_DEVICE_ENABLED_HIP)
    message(FATAL_ERROR "HIP feature is currently unavailable.")
endif()


if   (${COMPACK3D_ENABLE_DEVICE_AWARE_COMM})
    add_compile_definitions(COMPACK3D_DEVICE_COMM_ENABLED)
    set(tmp_var_device_comm "Device-to-Device")
endif()


message(NOTICE
    " COMPACK3D Configurations:\n"
    "     * Compile mode: ${CMAKE_BUILD_TYPE}\n"
    "     * Device programming model: ${tmp_var_device_prog_model}\n"
    "     * Device architecture: ${tmp_var_device_arch_label}\n"
    "     * Distributed communication between multiple devices: ${tmp_var_device_comm}\n"
    "     * Modules to be built:"
)
foreach(module_name ${tmp_var_modules})
    message(NOTICE "       - ${module_name}")
endforeach()


unset(tmp_var_modules)
unset(tmp_var_device_comm)
unset(tmp_var_device_arch_label)
unset(tmp_var_device_prog_model)
unset(tmp_var_device_count)


if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    add_compile_definitions(COMPACK3D_DEBUG_MODE_ENABLED)
endif()


########
### General C++ configurations
########
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CXX_WARNING_FLAGS -Wall -Wextra)

if    (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    string(APPEND CMAKE_CXX_FLAGS " -O3")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    string(APPEND CMAKE_CXX_FLAGS " -O0 -g")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Profiling")
    string(APPEND CMAKE_CXX_FLAGS " -O3 -g")
endif()



########
### General CUDA configurations
########
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDAToolkit REQUIRED)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
        set(CUDA_WARNING_FLAGS --Wext-lambda-captures-this --Wreorder)
        list(APPEND CXX_WARNING_FLAGS -Wno-pedantic)
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            list(APPEND CMAKE_CUDA_FLAGS -O0 -G --ptxas-options=-v)
        else()
            list(APPEND CMAKE_CUDA_FLAGS -O3)
            if (${CMAKE_BUILD_TYPE} STREQUAL "Profiling")
                list(APPEND CMAKE_CUDA_FLAGS -lineinfo)
            endif()
        endif()
        list(APPEND CMAKE_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr --forward-unknown-to-host-compiler)
        list(JOIN CMAKE_CUDA_FLAGS " " CMAKE_CUDA_FLAGS)
        message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
        message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
    else()
        message(FATAL_ERROR "Cannot find a valid CUDA compiler via CMAKE_CUDA_COMPILER")
    endif()
endif()



### Configure MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message(STATUS "MPI_CXX_COMPILER = ${MPI_CXX_COMPILER}")
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
endif()



add_subdirectory(src)

if (${COMPACK3D_BUILD_UNIT_TESTS})
    include(CTest)
    add_subdirectory(unit_tests)
endif()


if (${COMPACK3D_BUILD_BENCHMARK_TESTS})
    add_subdirectory(benchmarks)
endif()

if (${COMPACK3D_BUILD_STENCIL_APIS})
    add_subdirectory(stencils)
endif()

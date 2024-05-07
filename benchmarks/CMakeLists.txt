# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (benchmark_test_name "Local solve dimension-i")
    set (benchmark_test_src  Compack3D_benchmark_local_pcr.cpp)
    get_filename_component(exec_name ${benchmark_test_src} NAME_WE)
    add_executable(${exec_name} ${benchmark_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
endif()

# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (benchmark_test_name "Dist solve dimension-i")
    set (benchmark_test_src  Compack3D_benchmark_cuda_api_dim_i.cpp)
    get_filename_component(exec_name ${benchmark_test_src} NAME_WE)
    add_executable(${exec_name} ${benchmark_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
endif()

# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (benchmark_test_name "Dist solve dimension-j")
    set (benchmark_test_src  Compack3D_benchmark_cuda_api_dim_j.cpp)
    get_filename_component(exec_name ${benchmark_test_src} NAME_WE)
    add_executable(${exec_name} ${benchmark_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
endif()

# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (benchmark_test_name "Dist solve dimension-k")
    set (benchmark_test_src  Compack3D_benchmark_cuda_api_dim_k.cpp)
    get_filename_component(exec_name ${benchmark_test_src} NAME_WE)
    add_executable(${exec_name} ${benchmark_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
endif()

# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA} AND ${COMPACK3D_BUILD_STENCIL_APIS})
    set (benchmark_test_name "Explicit stencil kernels")
    set (benchmark_test_src  Compack3D_benchmark_kernels_stencil_7.cpp)
    get_filename_component(exec_name ${benchmark_test_src} NAME_WE)
    add_executable(${exec_name} ${benchmark_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D_stencils Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
endif()

# BEGIN WITH A NEW UNIT TEST
set (unit_test_name "Local factorization solve 3x3")
set (unit_test_src  unit_test_band_solve_3x3.cpp)
get_filename_component(exec_name ${unit_test_src} NAME_WE)
add_executable(${exec_name} ${unit_test_src})
target_link_libraries(${exec_name} PUBLIC Compack3D)
target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS})
add_test(NAME ${unit_test_name} COMMAND ${exec_name})


# BEGIN WITH A NEW UNIT TEST
set (unit_test_name "Local factorization solve 5x5")
set (unit_test_src  unit_test_band_solve_5x5.cpp)
get_filename_component(exec_name ${unit_test_src} NAME_WE)
add_executable(${exec_name} ${unit_test_src})
target_link_libraries(${exec_name} PUBLIC Compack3D)
target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS})
add_test(NAME ${unit_test_name} COMMAND ${exec_name})


# BEGIN WITH A NEW UNIT TEST
set (unit_test_name "Local factorization index")
set (unit_test_src  unit_test_local_fact_index.cpp)
get_filename_component(exec_name ${unit_test_src} NAME_WE)
add_executable(${exec_name} ${unit_test_src})
target_link_libraries(${exec_name} PUBLIC Compack3D)
target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS})
add_test(NAME ${unit_test_name} COMMAND ${exec_name})


# BEGIN WITH A NEW UNIT TEST
set (unit_test_name "Vanilla local solve using PCR for a penta-diagonal system")
set (unit_test_src  unit_test_local_fact_penta.cpp)
get_filename_component(exec_name ${unit_test_src} NAME_WE)
add_executable(${exec_name} ${unit_test_src})
target_link_libraries(${exec_name} PUBLIC Compack3D)
target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS})
add_test(NAME ${unit_test_name} COMMAND ${exec_name})


# BEGIN WITH A NEW UNIT TEST
set (unit_test_name "Vanilla local solve using PCR for a tridiagonal system")
set (unit_test_src  unit_test_local_fact_tri.cpp)
get_filename_component(exec_name ${unit_test_src} NAME_WE)
add_executable(${exec_name} ${unit_test_src})
target_link_libraries(${exec_name} PUBLIC Compack3D)
target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS})
add_test(NAME ${unit_test_name} COMMAND ${exec_name})


# BEGIN WITH A NEW UNIT TEST
set (unit_test_name "Vanilla local solve PCR batch")
set (unit_test_src  unit_test_vanilla_local_solve_batch.cpp)
get_filename_component(exec_name ${unit_test_src} NAME_WE)
add_executable(${exec_name} ${unit_test_src})
target_link_libraries(${exec_name} PUBLIC Compack3D)
target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS})
add_test(NAME ${unit_test_name} COMMAND ${exec_name})


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
set (unit_test_name "Vanilla distributed solve of a penta-diagonal system")
set (unit_test_src  unit_test_cuda_vanilla_dist_solve_penta.cpp)
get_filename_component(exec_name ${unit_test_src} NAME_WE)
add_executable(${exec_name} ${unit_test_src})
target_link_libraries(${exec_name} PUBLIC Compack3D)
target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS})
set_target_properties(${exec_name} PROPERTIES TEST_LAUNCHER "${COMPACK3D_MPI_EXEC_PREFIX}")
add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
set (unit_test_name "Vanilla distributed solve of a tridiagonal system")
set (unit_test_src  unit_test_cuda_vanilla_dist_solve_tri.cpp)
get_filename_component(exec_name ${unit_test_src} NAME_WE)
add_executable(${exec_name} ${unit_test_src})
target_link_libraries(${exec_name} PUBLIC Compack3D)
target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS})
set_target_properties(${exec_name} PROPERTIES TEST_LAUNCHER "${COMPACK3D_MPI_EXEC_PREFIX}")
add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (unit_test_name "Local solve of a penta-diagonal system in dimension-i")
    set (unit_test_src  unit_test_cuda_local_solve_penta_dim_i.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (unit_test_name "Local solve of a penta-diagonal system in dimension-j")
    set (unit_test_src  unit_test_cuda_local_solve_penta_dim_j.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (unit_test_name "Local solve of a penta-diagonal system in dimension-k")
    set (unit_test_src  unit_test_cuda_local_solve_penta_dim_k.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (unit_test_name "Local solve of a tridiagonal system in dimension-i")
    set (unit_test_src  unit_test_cuda_local_solve_tri_dim_i.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (unit_test_name "Local solve of a tridiagonal system in dimension-j")
    set (unit_test_src  unit_test_cuda_local_solve_tri_dim_j.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (unit_test_name "Local solve of a tridiagonal system in dimension-k")
    set (unit_test_src  unit_test_cuda_local_solve_tri_dim_k.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (unit_test_name "Copy slices from 3D array")
    set (unit_test_src  unit_test_cuda_utils_copy_slices.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA} AND ${COMPACK3D_ENABLE_DEVICE_AWARE_COMM})
    set (unit_test_name "Solve the 3D array in the i-dimension using MPI+CUDA model.")
    set (unit_test_src  unit_test_cuda_dist_solve_dim_i.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    set_target_properties(${exec_name} PROPERTIES TEST_LAUNCHER "${COMPACK3D_MPI_EXEC_PREFIX}")
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA} AND ${COMPACK3D_ENABLE_DEVICE_AWARE_COMM})
    set (unit_test_name "Solve the 3D array in the j-dimension using MPI+CUDA model.")
    set (unit_test_src  unit_test_cuda_dist_solve_dim_j.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    set_target_properties(${exec_name} PROPERTIES TEST_LAUNCHER "${COMPACK3D_MPI_EXEC_PREFIX}")
    add_test(NAME ${unit_test_name} COMMAND ${COMPACK3D_MPI_EXEC_PREFIX} ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA} AND ${COMPACK3D_ENABLE_DEVICE_AWARE_COMM})
    set (unit_test_name "Solve the 3D array in the k-dimension using MPI+CUDA model.")
    set (unit_test_src  unit_test_cuda_dist_solve_dim_k.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    set_target_properties(${exec_name} PROPERTIES TEST_LAUNCHER "${COMPACK3D_MPI_EXEC_PREFIX}")
    add_test(NAME ${unit_test_name} COMMAND ${COMPACK3D_MPI_EXEC_PREFIX} ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (unit_test_name "Solve the 3D array using wrapped penta-diagonal solver APIs in i-dimension")
    set (unit_test_src unit_test_cuda_penta_api_dim_i.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    set_target_properties(${exec_name} PROPERTIES TEST_LAUNCHER "${COMPACK3D_MPI_EXEC_PREFIX}")
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (unit_test_name "Solve the 3D array using wrapped penta-diagonal solver APIs in j-dimension")
    set (unit_test_src unit_test_cuda_penta_api_dim_j.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    set_target_properties(${exec_name} PROPERTIES TEST_LAUNCHER "${COMPACK3D_MPI_EXEC_PREFIX}")
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (unit_test_name "Solve the 3D array using wrapped penta-diagonal solver APIs in k-dimension")
    set (unit_test_src unit_test_cuda_penta_api_dim_k.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    set_target_properties(${exec_name} PROPERTIES TEST_LAUNCHER "${COMPACK3D_MPI_EXEC_PREFIX}")
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (unit_test_name "Solve the 3D array using wrapped tridiagonal solver APIs in i-dimension")
    set (unit_test_src unit_test_cuda_tri_api_dim_i.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    set_target_properties(${exec_name} PROPERTIES TEST_LAUNCHER "${COMPACK3D_MPI_EXEC_PREFIX}")
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (unit_test_name "Solve the 3D array using wrapped tridiagonal solver APIs in j-dimension")
    set (unit_test_src unit_test_cuda_tri_api_dim_j.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    set_target_properties(${exec_name} PROPERTIES TEST_LAUNCHER "${COMPACK3D_MPI_EXEC_PREFIX}")
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_ENABLE_DEVICE_CUDA})
    set (unit_test_name "Solve the 3D array using wrapped tridiagonal solver APIs in k-dimension")
    set (unit_test_src unit_test_cuda_tri_api_dim_k.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    set_target_properties(${exec_name} PROPERTIES TEST_LAUNCHER "${COMPACK3D_MPI_EXEC_PREFIX}")
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_BUILD_STENCIL_APIS})
    set (unit_test_name "Explicit 7-point stencil central difference in i-dimension")
    set (unit_test_src unit_test_explicit_central_diff_stencil_7_dim_i.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D_stencils Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_BUILD_STENCIL_APIS})
    set (unit_test_name "Explicit 7-point stencil central difference in j-dimension")
    set (unit_test_src unit_test_explicit_central_diff_stencil_7_dim_j.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D_stencils Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()


# BEGIN WITH A NEW UNIT TEST
if (${COMPACK3D_BUILD_STENCIL_APIS})
    set (unit_test_name "Explicit 7-point stencil central difference in k-dimension")
    set (unit_test_src unit_test_explicit_central_diff_stencil_7_dim_k.cpp)
    get_filename_component(exec_name ${unit_test_src} NAME_WE)
    add_executable(${exec_name} ${unit_test_src})
    target_link_libraries(${exec_name} PUBLIC Compack3D_stencils Compack3D)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas" ${CXX_WARNING_FLAGS} )
    add_test(NAME ${unit_test_name} COMMAND ${exec_name})
endif()

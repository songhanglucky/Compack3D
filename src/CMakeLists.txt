

set(COMPACK3D_RT_LIBS)

if (COMPACK3D_ENABLE_DEVICE_CUDA)
    set(COMPACK3D_RT_LIBS CUDA::cudart)
endif()


set(src_host
    Compack3D_utils.cpp
    Compack3D_penta.cpp
    Compack3D_tri.cpp
)

set(src_device
    Compack3D_utils_kernels.cu
    Compack3D_penta_kernels.cu
    Compack3D_tri_kernels.cu
)

set(src_api
    Compack3D_api.cpp    
)


add_library(Compack3D_host OBJECT ${src_host})
target_include_directories(Compack3D_host
    PRIVATE          ${CMAKE_CURRENT_SOURCE_DIR}
    SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(Compack3D_host PUBLIC ${MPI_CXX_LIBRARIES} ${COMPACK3D_RT_LIBS})


add_library(Compack3D_device OBJECT ${src_device})
target_include_directories(Compack3D_device
    PRIVATE          ${CMAKE_CURRENT_SOURCE_DIR}
    SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(Compack3D_device PUBLIC Compack3D_host ${MPI_CXX_LIBRARIES} ${COMPACK3D_RT_LIBS})


add_library(Compack3D_api ${src_api})
target_include_directories(Compack3D_api
    PRIVATE          ${CMAKE_CURRENT_SOURCE_DIR}
    SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(Compack3D_api PUBLIC Compack3D_device Compack3D_host ${MPI_CXX_LIBRARIES} ${COMPACK3D_RT_LIBS})


add_library(Compack3D $<TARGET_OBJECTS:Compack3D_host> $<TARGET_OBJECTS:Compack3D_device> $<TARGET_OBJECTS:Compack3D_api>)
target_include_directories(Compack3D SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Compack3D PUBLIC Compack3D_device Compack3D_host ${MPI_CXX_LIBRARIES} ${COMPACK3D_RT_LIBS})

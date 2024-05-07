

set(src_device
    Compack3D_stencils_kernels.cu
)


set(src_host
    Compack3D_stencils.cpp
)


add_library(Compack3D_stencil_device OBJECT ${src_device})
target_include_directories(Compack3D_stencil_device
    PRIVATE          ${CMAKE_CURRENT_SOURCE_DIR}
    SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(Compack3D_stencil_device PUBLIC Compack3D ${MPI_CXX_LIBRARIES} ${COMPACK3D_RT_LIBS})


add_library(Compack3D_stencils $<TARGET_OBJECTS:Compack3D_stencil_device>)
target_include_directories(Compack3D_stencils SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Compack3D_stencils PUBLIC Compack3D)

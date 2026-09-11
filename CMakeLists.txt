cmake_minimum_required(VERSION 3.21)
project(HypercubeCascade VERSION 1.0.4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-config generators (Ninja/Make): empty CMAKE_BUILD_TYPE disables
# $<CONFIG:…> flags below. Multi-config (VS, Xcode) leaves this alone.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- HypercubeCNN (vendored core; see third_party/HypercubeCNN/VENDORED.md) ---
add_subdirectory(third_party/HypercubeCNN)

# Shared compile / link flags (host core + demos)
add_library(cascade_compile_flags INTERFACE)
target_compile_options(cascade_compile_flags INTERFACE
    $<$<CONFIG:Release>:-O3 -march=x86-64-v2 -ffast-math>
    $<$<CONFIG:Debug>:-O0 -g>
    -Wall -Wextra -Wno-unknown-pragmas
)
if(MINGW)
    target_link_options(cascade_compile_flags INTERFACE
        -static-libgcc
        -static-libstdc++
    )
endif()

# Match host Release/Debug options on the vendored static library.
target_link_libraries(HypercubeCNNCore PRIVATE cascade_compile_flags)

# --- Core: Exciter + Reservoir + Readout + Cascade ---
add_library(HypercubeCascadeCore STATIC
    Exciter.cpp
    Reservoir.cpp
    Readout.cpp
    Cascade.cpp
)

target_include_directories(HypercubeCascadeCore
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

target_link_libraries(HypercubeCascadeCore
    PUBLIC  HypercubeCNNCore
    PRIVATE cascade_compile_flags
)

# --- Smoke executable ---
add_executable(HypercubeCascade
    main.cpp
)

target_link_libraries(HypercubeCascade
    PRIVATE
        HypercubeCascadeCore
        cascade_compile_flags
)

# Demos (only when this repo is the top-level project)
# Layout: examples/<demo>/…  shared headers: examples/common/
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    function(cascade_add_exe name)
        cmake_parse_arguments(CS "" "" "SOURCES;INCLUDES" ${ARGN})
        add_executable(${name} ${CS_SOURCES})
        target_link_libraries(${name} PRIVATE HypercubeCascadeCore cascade_compile_flags)
        target_include_directories(${name} PRIVATE
            ${PROJECT_SOURCE_DIR}/examples/common
            ${CS_INCLUDES}
        )
    endfunction()

    cascade_add_exe(cascade_synth SOURCES examples/synth/cascade_synth.cpp)
    cascade_add_exe(cascade_mnist
        SOURCES examples/mnist/cascade_mnist.cpp
        INCLUDES ${PROJECT_SOURCE_DIR}/examples/mnist)

    cascade_add_exe(cascade_raman
        SOURCES
            examples/RamanBaselineExtraction/cascade_raman.cpp
            examples/RamanBaselineExtraction/BaselineExtractor.cpp
            examples/RamanBaselineExtraction/RamanDataset.cpp
        INCLUDES ${PROJECT_SOURCE_DIR}/examples/RamanBaselineExtraction)
    cascade_add_exe(cascade_raman_extract
        SOURCES
            examples/RamanBaselineExtraction/cascade_raman_extract.cpp
            examples/RamanBaselineExtraction/BaselineExtractor.cpp
            examples/RamanBaselineExtraction/RamanDataset.cpp
            examples/RamanBaselineExtraction/RamanExtract.cpp
        INCLUDES ${PROJECT_SOURCE_DIR}/examples/RamanBaselineExtraction)
endif()

# Vendored HypercubeCNN — see VENDORED.md in this directory.
# Build glue for HypercubeCascade (not upstream). Core inherits host flags.

find_package(Threads REQUIRED)

add_library(HypercubeCNNCore STATIC
    HCNNConv.cpp
    HCNNPool.cpp
    HCNNNetwork.cpp
    HCNNReadout.cpp
    HCNN.cpp
    HCNNSpatialAug.cpp
    HCNNSpatialEmbed.cpp
    HCNNTrainHelpers.cpp
)
add_library(HypercubeCNN::HypercubeCNNCore ALIAS HypercubeCNNCore)

target_compile_features(HypercubeCNNCore PUBLIC cxx_std_23)
set_target_properties(HypercubeCNNCore PROPERTIES CXX_EXTENSIONS OFF)

target_include_directories(HypercubeCNNCore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(HypercubeCNNCore PUBLIC Threads::Threads)

target_compile_definitions(HypercubeCNNCore PRIVATE HCNN_FAST_TANH)

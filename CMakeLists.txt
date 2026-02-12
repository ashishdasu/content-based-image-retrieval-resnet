cmake_minimum_required(VERSION 3.10)
project(CBIR CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)

# Shared feature/distance library sources
set(CORE_SOURCES
    features.cpp
    distances.cpp
)

# Main CBIR query program
add_executable(cbir
    cbir.cpp
    ${CORE_SOURCES}
)
target_include_directories(cbir PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cbir PRIVATE ${OpenCV_LIBS})

# Offline feature pre-computation
add_executable(compute_features
    compute_features.cpp
    ${CORE_SOURCES}
)
target_include_directories(compute_features PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(compute_features PRIVATE ${OpenCV_LIBS})

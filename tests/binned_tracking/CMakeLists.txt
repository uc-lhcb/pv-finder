cmake_minimum_required(VERSION 3.11)

project(BinnedTracking LANGUAGES CXX)

find_package(ROOT CONFIG REQUIRED COMPONENTS Minuit)

include("${ROOT_USE_FILE}")

add_executable(test_tracking
    test_tracking.cc
    data.h
    hits.h
    tracklets.h
    trajectory.h
)

target_link_libraries(test_tracking ${ROOT_LIBRARIES})

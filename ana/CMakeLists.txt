cmake_minimum_required(VERSION 3.14)

project(PVFinderANA LANGUAGES CXX)

find_package(ROOT CONFIG REQUIRED)

# Looks like full address are not used in targets even in 6.18
include("${ROOT_USE_FILE}")

# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/Cellar/gperftools/2.9.1/lib/ -lprofiler")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/usr/local/Cellar/gperftools/2.9.1/lib/ -lprofiler")

file(GLOB_RECURSE
     HEADERS
     CONFIGURE_DEPENDS
     *.h
     )

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Header Files" FILES ${HEADERS})

add_library(core_lib
    ${HEADERS}
    makehist.C
    makehistfromtracks.C
    maketracks.C
    combinetruthvertices.C # emk addition
    )

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building with OpenMP")
    target_link_libraries(core_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(make_histogram_from_tracks
    apps/make_hist_from_tracks_main.C
    apps/CLI11.hpp
)

add_executable(make_histogram
    apps/make_hist_main.C
    apps/CLI11.hpp
)

add_executable(make_tracks
    apps/make_tracks_main.C
    apps/CLI11.hpp
)

# emk addition
add_executable(combine_truth_vertices
    apps/combine_truth_vertices_main.C
    apps/CLI11.hpp
)

target_link_libraries(core_lib PUBLIC
	ROOT::Physics
	ROOT::Core
	ROOT::MathCore
	ROOT::Tree
	ROOT::Matrix
	ROOT::Minuit
)

target_link_libraries(make_histogram PUBLIC core_lib)
target_link_libraries(make_tracks PUBLIC core_lib)
target_link_libraries(make_histogram_from_tracks PUBLIC core_lib)
target_link_libraries(combine_truth_vertices PUBLIC core_lib) # emk addition
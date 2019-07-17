cmake_minimum_required(VERSION 3.14)

project(PVFinderANA LANGUAGES CXX)

find_package(ROOT CONFIG REQUIRED)

# Looks like full address are not used in targets even in 6.18
include("${ROOT_USE_FILE}")

add_executable(make_histogram
    main.C
    makehist.C
    compute_over.h
    hit.h
    hits.h
    lhcbStyle.h
    point.h
    property.h
    tracks.h
    trajectory.h
    triplet.h
    triplet_utils.h
    fcn.h
    data/raw_hits.h
    data/raw_kernel.h
    data/raw_tracks.h
)

target_link_libraries(make_histogram PUBLIC
	ROOT::Physics
	ROOT::Core
	ROOT::MathCore
	ROOT::Tree
	ROOT::Matrix
	ROOT::Minuit
)

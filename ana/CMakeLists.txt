cmake_minimum_required(VERSION 3.14)

project(PVFinderANA LANGUAGES CXX)

find_package(ROOT CONFIG REQUIRED)

add_executable(make_histogram
    main.C
    makehist.C
    data.h
    hit.h
    hits.h
    lhcbStyle.h
    point.h
    property.h
    tracks.h
    trajectory.h
    triplet.h
    triplet_utils.h
    utils.h
)

target_link_libraries(make_histogram PUBLIC
	ROOT::Physics
	ROOT::Core
	ROOT::MathCore
	ROOT::Tree
	ROOT::Matrix
	ROOT::Minuit
)

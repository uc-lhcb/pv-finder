cmake_minimum_required(VERSION 3.14)

project(PVFinderANA LANGUAGES CXX)

find_package(ROOT CONFIG REQUIRED)

add_executable(make_histogram
    data.h
    hits.h
    lhcbStyle.h
    makehist.C
    point.h
    runall.sh
    testA.py
    testAna.py
    tracks.h
    trajectory.h
    triplet.h
    utils.h
    main.C
)

target_link_libraries(make_histogram PUBLIC
	ROOT::Physics
	ROOT::Core
	ROOT::MathCore
	ROOT::Tree
	ROOT::Matrix
	ROOT::Minuit
)

cmake_minimum_required(VERSION 3.14)

project(PVFinderANA LANGUAGES CXX)

find_package(ROOT CONFIG REQUIRED)

# Looks like full address are not used in targets even in 6.18
include("${ROOT_USE_FILE}")

file(GLOB HEADER_FILES CONFIGURE_DEPENDS *.h data/*.h)
set(SRC_FILES main.C makehist.C)

add_executable(make_histogram ${SRC_FILES} ${HEADER_FILES})

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/" PREFIX "Header Files" FILES ${HEADER_FILES})

target_link_libraries(make_histogram PUBLIC
	ROOT::Physics
	ROOT::Core
	ROOT::MathCore
	ROOT::Tree
	ROOT::Matrix
	ROOT::Minuit
)

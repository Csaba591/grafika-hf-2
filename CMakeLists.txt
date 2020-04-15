cmake_minimum_required(VERSION 3.7)
project(openglh_clion)

#a lot of gcc warning flags
SET(CMAKE_CXX_FLAGS "-std=c++11 -Wpedantic -Wall -Wextra -Wdouble-promotion -Wformat=2 -Wformat-overflow=2 -Wformat-signedness -Wformat-truncation=2 -Wnull-dereference -Winit-self -Wmissing-include-dirs -Wmissing-braces -Wshift-overflow=2 -Wswitch-default -Wswitch-enum -Wsync-nand -Wunused -Wuninitialized -Wstrict-overflow=4 -Wsuggest-attribute=pure -Wsuggest-attribute=const -Wsuggest-attribute=noreturn -Wsuggest-attribute=format -Wsuggest-final-types -Wsuggest-final-methods -Wsuggest-override -Walloc-zero -Wduplicated-branches -Wduplicated-cond -Wframe-address -Wfloat-equal -Wshadow -Wunsafe-loop-optimizations -Waligned-new=all -Wplacement-new=2 -Wundef -Wunused-macros -Wconditionally-supported -Wconversion -Wzero-as-null-pointer-constant -Wuseless-cast -Wsign-conversion -Wsized-deallocation -Wlogical-op -Wnormalized -Wpacked -Wredundant-decls -Wrestrict -Winvalid-pch")

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin) #binary dir to <root_directory>/bin

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}) #executable output to <root_directory>/bin
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}) #library output to <root_directory>/bin

#source files (framework projects: 3dendzsinke, bezier, beziersurface, clifford, complex, imageviewer, pixeldriven2dgraphics, pman, quaternion, raytrace, raytrace_0, rootfinder, Skeleton, smoothtriangle, texture)
set(SOURCE_FILES framework.h framework.cpp Skeleton.cpp)
#set(SOURCE_FILES framework.h framework.cpp bezier.cpp) #source files (space game)
#set(SOURCE_FILES fractal.cpp) #set source files (non-framework projects: greentriangle, julia, fractal, pathtracingfinal)


include_directories(include) #adds <root_directory>/include to the include directories
link_directories(lib) #adds <root_directory>/lib to the link directories

add_executable(${PROJECT_NAME} ${SOURCE_FILES}) #creates executable from source files

target_link_libraries(${PROJECT_NAME} freeglut glew32 opengl32) #adds libraries for linking
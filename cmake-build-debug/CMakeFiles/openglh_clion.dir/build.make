# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "D:\Programs\CLion 2019.3.2\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "D:\Programs\CLion 2019.3.2\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\Documents\Grafika\hf2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\Documents\Grafika\hf2\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/openglh_clion.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/openglh_clion.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/openglh_clion.dir/flags.make

CMakeFiles/openglh_clion.dir/framework.cpp.obj: CMakeFiles/openglh_clion.dir/flags.make
CMakeFiles/openglh_clion.dir/framework.cpp.obj: CMakeFiles/openglh_clion.dir/includes_CXX.rsp
CMakeFiles/openglh_clion.dir/framework.cpp.obj: ../framework.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Documents\Grafika\hf2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/openglh_clion.dir/framework.cpp.obj"
	D:\Programs\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\openglh_clion.dir\framework.cpp.obj -c D:\Documents\Grafika\hf2\framework.cpp

CMakeFiles/openglh_clion.dir/framework.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openglh_clion.dir/framework.cpp.i"
	D:\Programs\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Documents\Grafika\hf2\framework.cpp > CMakeFiles\openglh_clion.dir\framework.cpp.i

CMakeFiles/openglh_clion.dir/framework.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openglh_clion.dir/framework.cpp.s"
	D:\Programs\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\Documents\Grafika\hf2\framework.cpp -o CMakeFiles\openglh_clion.dir\framework.cpp.s

CMakeFiles/openglh_clion.dir/Skeleton.cpp.obj: CMakeFiles/openglh_clion.dir/flags.make
CMakeFiles/openglh_clion.dir/Skeleton.cpp.obj: CMakeFiles/openglh_clion.dir/includes_CXX.rsp
CMakeFiles/openglh_clion.dir/Skeleton.cpp.obj: ../Skeleton.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\Documents\Grafika\hf2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/openglh_clion.dir/Skeleton.cpp.obj"
	D:\Programs\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\openglh_clion.dir\Skeleton.cpp.obj -c D:\Documents\Grafika\hf2\Skeleton.cpp

CMakeFiles/openglh_clion.dir/Skeleton.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openglh_clion.dir/Skeleton.cpp.i"
	D:\Programs\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Documents\Grafika\hf2\Skeleton.cpp > CMakeFiles\openglh_clion.dir\Skeleton.cpp.i

CMakeFiles/openglh_clion.dir/Skeleton.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openglh_clion.dir/Skeleton.cpp.s"
	D:\Programs\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\Documents\Grafika\hf2\Skeleton.cpp -o CMakeFiles\openglh_clion.dir\Skeleton.cpp.s

# Object files for target openglh_clion
openglh_clion_OBJECTS = \
"CMakeFiles/openglh_clion.dir/framework.cpp.obj" \
"CMakeFiles/openglh_clion.dir/Skeleton.cpp.obj"

# External object files for target openglh_clion
openglh_clion_EXTERNAL_OBJECTS =

../bin/openglh_clion.exe: CMakeFiles/openglh_clion.dir/framework.cpp.obj
../bin/openglh_clion.exe: CMakeFiles/openglh_clion.dir/Skeleton.cpp.obj
../bin/openglh_clion.exe: CMakeFiles/openglh_clion.dir/build.make
../bin/openglh_clion.exe: CMakeFiles/openglh_clion.dir/linklibs.rsp
../bin/openglh_clion.exe: CMakeFiles/openglh_clion.dir/objects1.rsp
../bin/openglh_clion.exe: CMakeFiles/openglh_clion.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\Documents\Grafika\hf2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ..\bin\openglh_clion.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\openglh_clion.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/openglh_clion.dir/build: ../bin/openglh_clion.exe

.PHONY : CMakeFiles/openglh_clion.dir/build

CMakeFiles/openglh_clion.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\openglh_clion.dir\cmake_clean.cmake
.PHONY : CMakeFiles/openglh_clion.dir/clean

CMakeFiles/openglh_clion.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\Documents\Grafika\hf2 D:\Documents\Grafika\hf2 D:\Documents\Grafika\hf2\cmake-build-debug D:\Documents\Grafika\hf2\cmake-build-debug D:\Documents\Grafika\hf2\cmake-build-debug\CMakeFiles\openglh_clion.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/openglh_clion.dir/depend


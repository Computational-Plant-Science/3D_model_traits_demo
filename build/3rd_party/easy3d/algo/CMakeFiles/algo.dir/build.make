# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/suxing/AdTree

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/suxing/AdTree/build

# Include any dependencies generated for this target.
include 3rd_party/easy3d/algo/CMakeFiles/algo.dir/depend.make

# Include the progress variables for this target.
include 3rd_party/easy3d/algo/CMakeFiles/algo.dir/progress.make

# Include the compile flags for this target's objects.
include 3rd_party/easy3d/algo/CMakeFiles/algo.dir/flags.make

3rd_party/easy3d/algo/CMakeFiles/algo.dir/remove_duplication.cpp.o: 3rd_party/easy3d/algo/CMakeFiles/algo.dir/flags.make
3rd_party/easy3d/algo/CMakeFiles/algo.dir/remove_duplication.cpp.o: ../3rd_party/easy3d/algo/remove_duplication.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/suxing/AdTree/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object 3rd_party/easy3d/algo/CMakeFiles/algo.dir/remove_duplication.cpp.o"
	cd /home/suxing/AdTree/build/3rd_party/easy3d/algo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/algo.dir/remove_duplication.cpp.o -c /home/suxing/AdTree/3rd_party/easy3d/algo/remove_duplication.cpp

3rd_party/easy3d/algo/CMakeFiles/algo.dir/remove_duplication.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/algo.dir/remove_duplication.cpp.i"
	cd /home/suxing/AdTree/build/3rd_party/easy3d/algo && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/suxing/AdTree/3rd_party/easy3d/algo/remove_duplication.cpp > CMakeFiles/algo.dir/remove_duplication.cpp.i

3rd_party/easy3d/algo/CMakeFiles/algo.dir/remove_duplication.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/algo.dir/remove_duplication.cpp.s"
	cd /home/suxing/AdTree/build/3rd_party/easy3d/algo && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/suxing/AdTree/3rd_party/easy3d/algo/remove_duplication.cpp -o CMakeFiles/algo.dir/remove_duplication.cpp.s

# Object files for target algo
algo_OBJECTS = \
"CMakeFiles/algo.dir/remove_duplication.cpp.o"

# External object files for target algo
algo_EXTERNAL_OBJECTS =

lib/libalgo.a: 3rd_party/easy3d/algo/CMakeFiles/algo.dir/remove_duplication.cpp.o
lib/libalgo.a: 3rd_party/easy3d/algo/CMakeFiles/algo.dir/build.make
lib/libalgo.a: 3rd_party/easy3d/algo/CMakeFiles/algo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/suxing/AdTree/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../../../lib/libalgo.a"
	cd /home/suxing/AdTree/build/3rd_party/easy3d/algo && $(CMAKE_COMMAND) -P CMakeFiles/algo.dir/cmake_clean_target.cmake
	cd /home/suxing/AdTree/build/3rd_party/easy3d/algo && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/algo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
3rd_party/easy3d/algo/CMakeFiles/algo.dir/build: lib/libalgo.a

.PHONY : 3rd_party/easy3d/algo/CMakeFiles/algo.dir/build

3rd_party/easy3d/algo/CMakeFiles/algo.dir/clean:
	cd /home/suxing/AdTree/build/3rd_party/easy3d/algo && $(CMAKE_COMMAND) -P CMakeFiles/algo.dir/cmake_clean.cmake
.PHONY : 3rd_party/easy3d/algo/CMakeFiles/algo.dir/clean

3rd_party/easy3d/algo/CMakeFiles/algo.dir/depend:
	cd /home/suxing/AdTree/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/suxing/AdTree /home/suxing/AdTree/3rd_party/easy3d/algo /home/suxing/AdTree/build /home/suxing/AdTree/build/3rd_party/easy3d/algo /home/suxing/AdTree/build/3rd_party/easy3d/algo/CMakeFiles/algo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 3rd_party/easy3d/algo/CMakeFiles/algo.dir/depend

# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/suxing/AdTree

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/suxing/AdTree/Release

# Include any dependencies generated for this target.
include 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/compiler_depend.make

# Include the progress variables for this target.
include 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/progress.make

# Include the compile flags for this target's objects.
include 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/flags.make

3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.o: 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/flags.make
3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.o: ../3rd_party/tinyfiledialogs/tinyfiledialogs.c
3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.o: 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/suxing/AdTree/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.o"
	cd /home/suxing/AdTree/Release/3rd_party/tinyfiledialogs && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.o -MF CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.o.d -o CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.o -c /home/suxing/AdTree/3rd_party/tinyfiledialogs/tinyfiledialogs.c

3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.i"
	cd /home/suxing/AdTree/Release/3rd_party/tinyfiledialogs && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/suxing/AdTree/3rd_party/tinyfiledialogs/tinyfiledialogs.c > CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.i

3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.s"
	cd /home/suxing/AdTree/Release/3rd_party/tinyfiledialogs && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/suxing/AdTree/3rd_party/tinyfiledialogs/tinyfiledialogs.c -o CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.s

# Object files for target 3rd_tinyfiledialogs
3rd_tinyfiledialogs_OBJECTS = \
"CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.o"

# External object files for target 3rd_tinyfiledialogs
3rd_tinyfiledialogs_EXTERNAL_OBJECTS =

lib/lib3rd_tinyfiledialogs.a: 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/tinyfiledialogs.c.o
lib/lib3rd_tinyfiledialogs.a: 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/build.make
lib/lib3rd_tinyfiledialogs.a: 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/suxing/AdTree/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library ../../lib/lib3rd_tinyfiledialogs.a"
	cd /home/suxing/AdTree/Release/3rd_party/tinyfiledialogs && $(CMAKE_COMMAND) -P CMakeFiles/3rd_tinyfiledialogs.dir/cmake_clean_target.cmake
	cd /home/suxing/AdTree/Release/3rd_party/tinyfiledialogs && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/3rd_tinyfiledialogs.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/build: lib/lib3rd_tinyfiledialogs.a
.PHONY : 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/build

3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/clean:
	cd /home/suxing/AdTree/Release/3rd_party/tinyfiledialogs && $(CMAKE_COMMAND) -P CMakeFiles/3rd_tinyfiledialogs.dir/cmake_clean.cmake
.PHONY : 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/clean

3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/depend:
	cd /home/suxing/AdTree/Release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/suxing/AdTree /home/suxing/AdTree/3rd_party/tinyfiledialogs /home/suxing/AdTree/Release /home/suxing/AdTree/Release/3rd_party/tinyfiledialogs /home/suxing/AdTree/Release/3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 3rd_party/tinyfiledialogs/CMakeFiles/3rd_tinyfiledialogs.dir/depend

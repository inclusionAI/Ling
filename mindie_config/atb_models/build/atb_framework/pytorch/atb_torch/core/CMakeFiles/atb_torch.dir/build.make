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
CMAKE_SOURCE_DIR = /home/zhaozixin/MindIE-LLM/examples/atb_models

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhaozixin/MindIE-LLM/examples/atb_models/build

# Include any dependencies generated for this target.
include atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/compiler_depend.make

# Include the progress variables for this target.
include atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/progress.make

# Include the compile flags for this target's objects.
include atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/flags.make

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/flags.make
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.o: ../atb_framework/pytorch/atb_torch/core/base/atb_context_factory.cpp
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaozixin/MindIE-LLM/examples/atb_models/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.o"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.o -MF CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.o.d -o CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.o -c /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/atb_context_factory.cpp

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.i"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/atb_context_factory.cpp > CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.i

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.s"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/atb_context_factory.cpp -o CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.s

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/base_operation.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/flags.make
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/base_operation.cpp.o: ../atb_framework/pytorch/atb_torch/core/base/base_operation.cpp
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/base_operation.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaozixin/MindIE-LLM/examples/atb_models/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/base_operation.cpp.o"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/base_operation.cpp.o -MF CMakeFiles/atb_torch.dir/base/base_operation.cpp.o.d -o CMakeFiles/atb_torch.dir/base/base_operation.cpp.o -c /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/base_operation.cpp

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/base_operation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/atb_torch.dir/base/base_operation.cpp.i"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/base_operation.cpp > CMakeFiles/atb_torch.dir/base/base_operation.cpp.i

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/base_operation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/atb_torch.dir/base/base_operation.cpp.s"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/base_operation.cpp -o CMakeFiles/atb_torch.dir/base/base_operation.cpp.s

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/config.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/flags.make
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/config.cpp.o: ../atb_framework/pytorch/atb_torch/core/base/config.cpp
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/config.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaozixin/MindIE-LLM/examples/atb_models/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/config.cpp.o"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/config.cpp.o -MF CMakeFiles/atb_torch.dir/base/config.cpp.o.d -o CMakeFiles/atb_torch.dir/base/config.cpp.o -c /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/config.cpp

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/atb_torch.dir/base/config.cpp.i"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/config.cpp > CMakeFiles/atb_torch.dir/base/config.cpp.i

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/atb_torch.dir/base/config.cpp.s"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/config.cpp -o CMakeFiles/atb_torch.dir/base/config.cpp.s

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/graph_operation.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/flags.make
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/graph_operation.cpp.o: ../atb_framework/pytorch/atb_torch/core/base/graph_operation.cpp
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/graph_operation.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaozixin/MindIE-LLM/examples/atb_models/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/graph_operation.cpp.o"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/graph_operation.cpp.o -MF CMakeFiles/atb_torch.dir/base/graph_operation.cpp.o.d -o CMakeFiles/atb_torch.dir/base/graph_operation.cpp.o -c /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/graph_operation.cpp

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/graph_operation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/atb_torch.dir/base/graph_operation.cpp.i"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/graph_operation.cpp > CMakeFiles/atb_torch.dir/base/graph_operation.cpp.i

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/graph_operation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/atb_torch.dir/base/graph_operation.cpp.s"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/graph_operation.cpp -o CMakeFiles/atb_torch.dir/base/graph_operation.cpp.s

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/flags.make
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation.cpp.o: ../atb_framework/pytorch/atb_torch/core/base/operation.cpp
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaozixin/MindIE-LLM/examples/atb_models/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation.cpp.o"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation.cpp.o -MF CMakeFiles/atb_torch.dir/base/operation.cpp.o.d -o CMakeFiles/atb_torch.dir/base/operation.cpp.o -c /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/operation.cpp

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/atb_torch.dir/base/operation.cpp.i"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/operation.cpp > CMakeFiles/atb_torch.dir/base/operation.cpp.i

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/atb_torch.dir/base/operation.cpp.s"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/operation.cpp -o CMakeFiles/atb_torch.dir/base/operation.cpp.s

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation_factory.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/flags.make
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation_factory.cpp.o: ../atb_framework/pytorch/atb_torch/core/base/operation_factory.cpp
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation_factory.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaozixin/MindIE-LLM/examples/atb_models/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation_factory.cpp.o"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation_factory.cpp.o -MF CMakeFiles/atb_torch.dir/base/operation_factory.cpp.o.d -o CMakeFiles/atb_torch.dir/base/operation_factory.cpp.o -c /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/operation_factory.cpp

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation_factory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/atb_torch.dir/base/operation_factory.cpp.i"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/operation_factory.cpp > CMakeFiles/atb_torch.dir/base/operation_factory.cpp.i

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation_factory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/atb_torch.dir/base/operation_factory.cpp.s"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/operation_factory.cpp -o CMakeFiles/atb_torch.dir/base/operation_factory.cpp.s

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/utils.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/flags.make
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/utils.cpp.o: ../atb_framework/pytorch/atb_torch/core/base/utils.cpp
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/utils.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaozixin/MindIE-LLM/examples/atb_models/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/utils.cpp.o"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/utils.cpp.o -MF CMakeFiles/atb_torch.dir/base/utils.cpp.o.d -o CMakeFiles/atb_torch.dir/base/utils.cpp.o -c /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/utils.cpp

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/atb_torch.dir/base/utils.cpp.i"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/utils.cpp > CMakeFiles/atb_torch.dir/base/utils.cpp.i

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/atb_torch.dir/base/utils.cpp.s"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/base/utils.cpp -o CMakeFiles/atb_torch.dir/base/utils.cpp.s

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/flags.make
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.o: ../atb_framework/pytorch/atb_torch/core/operation_register/operation_register.cpp
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaozixin/MindIE-LLM/examples/atb_models/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.o"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.o -MF CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.o.d -o CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.o -c /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/operation_register/operation_register.cpp

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.i"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/operation_register/operation_register.cpp > CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.i

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.s"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/operation_register/operation_register.cpp -o CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.s

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/flags.make
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.o: ../atb_framework/pytorch/atb_torch/core/pybind/pybinds.cpp
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.o: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaozixin/MindIE-LLM/examples/atb_models/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.o"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.o -MF CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.o.d -o CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.o -c /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/pybind/pybinds.cpp

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.i"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/pybind/pybinds.cpp > CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.i

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.s"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core/pybind/pybinds.cpp -o CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.s

# Object files for target atb_torch
atb_torch_OBJECTS = \
"CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.o" \
"CMakeFiles/atb_torch.dir/base/base_operation.cpp.o" \
"CMakeFiles/atb_torch.dir/base/config.cpp.o" \
"CMakeFiles/atb_torch.dir/base/graph_operation.cpp.o" \
"CMakeFiles/atb_torch.dir/base/operation.cpp.o" \
"CMakeFiles/atb_torch.dir/base/operation_factory.cpp.o" \
"CMakeFiles/atb_torch.dir/base/utils.cpp.o" \
"CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.o" \
"CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.o"

# External object files for target atb_torch
atb_torch_EXTERNAL_OBJECTS =

atb_framework/pytorch/atb_torch/core/_libatb_torch.so: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/atb_context_factory.cpp.o
atb_framework/pytorch/atb_torch/core/_libatb_torch.so: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/base_operation.cpp.o
atb_framework/pytorch/atb_torch/core/_libatb_torch.so: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/config.cpp.o
atb_framework/pytorch/atb_torch/core/_libatb_torch.so: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/graph_operation.cpp.o
atb_framework/pytorch/atb_torch/core/_libatb_torch.so: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation.cpp.o
atb_framework/pytorch/atb_torch/core/_libatb_torch.so: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/operation_factory.cpp.o
atb_framework/pytorch/atb_torch/core/_libatb_torch.so: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/base/utils.cpp.o
atb_framework/pytorch/atb_torch/core/_libatb_torch.so: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/operation_register/operation_register.cpp.o
atb_framework/pytorch/atb_torch/core/_libatb_torch.so: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/pybind/pybinds.cpp.o
atb_framework/pytorch/atb_torch/core/_libatb_torch.so: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/build.make
atb_framework/pytorch/atb_torch/core/_libatb_torch.so: atb_framework/core/libatb_speed_core.so
atb_framework/pytorch/atb_torch/core/_libatb_torch.so: atb_framework/operations/libatb_speed_operations.so
atb_framework/pytorch/atb_torch/core/_libatb_torch.so: atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhaozixin/MindIE-LLM/examples/atb_models/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX shared library _libatb_torch.so"
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/atb_torch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/build: atb_framework/pytorch/atb_torch/core/_libatb_torch.so
.PHONY : atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/build

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/clean:
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core && $(CMAKE_COMMAND) -P CMakeFiles/atb_torch.dir/cmake_clean.cmake
.PHONY : atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/clean

atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/depend:
	cd /home/zhaozixin/MindIE-LLM/examples/atb_models/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhaozixin/MindIE-LLM/examples/atb_models /home/zhaozixin/MindIE-LLM/examples/atb_models/atb_framework/pytorch/atb_torch/core /home/zhaozixin/MindIE-LLM/examples/atb_models/build /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core /home/zhaozixin/MindIE-LLM/examples/atb_models/build/atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : atb_framework/pytorch/atb_torch/core/CMakeFiles/atb_torch.dir/depend


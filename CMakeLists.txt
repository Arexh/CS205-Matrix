cmake_minimum_required(VERSION 3.14)

project(CS205-Matrix)

# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 20)

set(TestModule matrixTest)

find_package(OpenCV REQUIRED)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${GTEST_INCLUDE_DIRS})

aux_source_directory(./src DIR_SRCS)

include_directories(include)

add_executable(main 
  ${DIR_SRCS}
  include/Matrix.h
)

target_link_libraries(main ${OpenCV_LIBS})

# TEST
enable_testing()

aux_source_directory(./test DIR_TESTS)

add_executable(
  ${TestModule}
  ${DIR_TESTS}
)
target_link_libraries(
  ${TestModule}
  gtest_main
  ${OpenCV_LIBS}
)

include(GoogleTest)
gtest_discover_tests(${TestModule})
project(pyssdr)

cmake_minimum_required(VERSION 3.15)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CONFIGURATION_TYPES "Release;" CACHE STRING "" FORCE)

include_directories("${PROJECT_SOURCE_DIR}/include")
find_package(Eigen REQUIRED)
include_directories("${EIGEN_INCLUDE_DIR}")

message(STATUS "Eigen dir: ${EIGEN_INCLUDE_DIR}")


if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}" CACHE PATH "..." FORCE)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(FBXSDK)
include_directories("${FBXSDK_INCLUDE_DIR}")
link_libraries("${FBXSDK_LIBS}")

add_subdirectory(pybind11)

link_libraries("-lxml2 -lz -ldl -L/usr/local/lib -lImath -lHalf -lIex -lIexMath -lIlmThread -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -pthread -I/usr/local/include/OpenEXR")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -D_GLIBCXX_USE_CXX11_ABI=0")

file(GLOB CMD_SOURCE
	"src/*.h"
	"src/*.cpp"
)

pybind11_add_module(pyssdr "${CMD_SOURCE}")




# add_executable(DemBonesCmd "${CMD_SOURCE}")
# set_target_properties(DemBonesCmd PROPERTIES OUTPUT_NAME "DemBones")



# target_include_directories(DemBonesCmd PUBLIC
# 	"${TCLAP_INCLUDE_DIR}"
# 	"${FBXSDK_INCLUDE_DIR}"
# 	"${ALEMBIC_INCLUDE_DIR}"
# 	"${Imath_INCLUDE_DIR}"
# )

# target_link_libraries(DemBonesCmd PUBLIC 
# 	"${FBXSDK_LIBS}"
# 	"${ALEMBIC_LIBS}"
# )

# if(WIN32)
# 	set_target_properties(DemBonesCmd PROPERTIES LINK_FLAGS "/force:multiple")
# 	install(TARGETS DemBonesCmd DESTINATION "bin/Windows")
# elseif(APPLE)
# 	set_target_properties(DemBonesCmd PROPERTIES LINK_FLAGS "-Wl,-F/Library/Frameworks")
# 	target_link_libraries(DemBonesCmd PUBLIC "-framework CoreFoundation -lxml2 -lz -liconv")
# 	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wno-switch -Wno-return-stack-address")
# 	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-switch -Wno-return-stack-address")
# 	install(TARGETS DemBonesCmd DESTINATION "bin/MacOS")
# else()
# 	install(TARGETS DemBonesCmd DESTINATION "bin/Linux")
# endif()


# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.14)

#-------------------------------------------------------------------------------
# PROJECT configuration
#-------------------------------------------------------------------------------
set(Library_NAME vCUDA CACHE STRING "The name of the main library.")
set(component_NAME core)

string(TOLOWER ${Library_NAME}   library_NAME)
string(TOUPPER ${Library_NAME}   LIBRARY_NAME)
string(TOUPPER ${component_NAME} COMPONENT_NAME)

project(${component_NAME}
  LANGUAGES CXX)

#-------------------------------------------------------------------------------
# LIBRARY configuration
#-------------------------------------------------------------------------------
add_library(${PROJECT_NAME} OBJECT)
add_library(${Library_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME}
  PRIVATE # these are only included below so that they appear when using IDEs
          include/vcuda.h include/vcuda/auto_cast.h
          include/vcuda/core/nullstream.h)

target_include_directories(${PROJECT_NAME}
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>)

#-------------------------------------------------------------------------------
# OPTIONS configuration
#-------------------------------------------------------------------------------
option(VCUDA_WITH_LOGGING "Enable diagnostic logging." OFF)

target_compile_definitions(${PROJECT_NAME}
  PUBLIC $<$<BOOL:${VCUDA_WITH_LOGGING}>:VCUDA_WITH_LOGGING>)

#-------------------------------------------------------------------------------
# EXPORT configuration
#-------------------------------------------------------------------------------
include(GenerateExportHeader)

generate_export_header(${PROJECT_NAME}
  PREFIX_NAME      ${LIBRARY_NAME}_
  EXPORT_FILE_NAME include/${library_NAME}/${component_NAME}/export.h)

target_compile_definitions(${PROJECT_NAME}
  PUBLIC $<IF:$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>,
          ${LIBRARY_NAME}_${COMPONENT_NAME}_STATIC_DEFINE,>)

#-------------------------------------------------------------------------------
# TEST configuration
#-------------------------------------------------------------------------------
#if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#  include(CTest)
#  if(BUILD_TESTING)
#    add_subdirectory(test)
#  endif()
#endif()

#-------------------------------------------------------------------------------
# INSTALL configuration
#-------------------------------------------------------------------------------
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

configure_package_config_file(${component_NAME}-config.cmake.in
                      cmake/${component_NAME}-config.cmake
  INSTALL_DESTINATION lib/cmake/${library_NAME})

install(TARGETS ${PROJECT_NAME}
  EXPORT   ${PROJECT_NAME}Targets
  RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
           COMPONENT   ${Library_NAME}_Runtime
  LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
           COMPONENT          ${Library_NAME}_Runtime
           NAMELINK_COMPONENT ${Library_NAME}_Development
  ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
           COMPONENT   ${Library_NAME}_Development
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ${PROJECT_NAME}Targets
  FILE        ${component_NAME}-targets.cmake
  NAMESPACE   ${Library_NAME}::
  DESTINATION lib/cmake/${library_NAME}
  COMPONENT   ${Library_NAME}_Development)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
                  ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT   ${Library_NAME}_Development)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cmake/
  DESTINATION lib/cmake/${library_NAME}
  COMPONENT   ${Library_NAME}_Development)

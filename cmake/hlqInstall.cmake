include_guard(GLOBAL)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS hlq EXPORT hlqTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT hlqTargets NAMESPACE hlq:: DESTINATION ${CMAKE_INSTALL_DATADIR}/hlq/cmake)

configure_package_config_file(cmake/hlqConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/hlqConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/hlq/cmake)
# 0.x releases may break the interface between minor versions.
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/hlqConfigVersion.cmake
    COMPATIBILITY SameMinorVersion
    ARCH_INDEPENDENT)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hlqConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/hlqConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_DATADIR}/hlq/cmake)
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/hlq)

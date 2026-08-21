cmake_minimum_required(VERSION 3.20)
project(MiniMaxBrainNative VERSION 0.3.1 LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MMB_BUILD_TESTS "Build native MiniMaxBrain tests" ON)
option(MMB_WITH_LLAMA "Build against the pinned llama.cpp source" ON)

if(MMB_WITH_LLAMA)
    set(MMB_LLAMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp" CACHE PATH "Pinned llama.cpp source directory")
    if(NOT EXISTS "${MMB_LLAMA_SOURCE_DIR}/include/llama.h")
        message(FATAL_ERROR "Pinned llama.cpp source not found at ${MMB_LLAMA_SOURCE_DIR}")
    endif()

    # MiniMaxBrain owns the llama.cpp build. Keep the embedded dependency small
    # and deterministic; user-facing tools are built separately when needed.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build embedded llama.cpp statically" FORCE)
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_RPC OFF CACHE BOOL "" FORCE)

    add_subdirectory("${MMB_LLAMA_SOURCE_DIR}" llama EXCLUDE_FROM_ALL)
endif()

add_library(mmb_core STATIC
    src/mmb_model.cpp
    src/mmb_cache.cpp
    src/mmb_pager.cpp
    src/mmb_kernel.cpp
    src/mmb_sha256.cpp
)
target_include_directories(mmb_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(mmb_core PUBLIC cxx_std_20)
set_target_properties(mmb_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MMB_WITH_LLAMA)
    target_sources(mmb_core PRIVATE
        src/mmb_ggml_bridge.cpp
        src/mmb_llama_loader.cpp
        src/mmb_runtime.cpp
    )
    target_link_libraries(mmb_core PUBLIC llama ggml ggml-cpu)
    target_compile_definitions(mmb_core PUBLIC MMB_WITH_LLAMA=1)
endif()

add_library(mmb_backend SHARED src/mmb_c_api.cpp)
target_include_directories(mmb_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(mmb_backend PUBLIC cxx_std_20)
target_compile_definitions(mmb_backend PRIVATE MMB_BACKEND_BUILD=1)
if(MMB_WITH_LLAMA)
    target_compile_definitions(mmb_backend PRIVATE MMB_WITH_LLAMA=1)
endif()
target_link_libraries(mmb_backend PRIVATE mmb_core)

if(MSVC)
    target_compile_options(mmb_core PRIVATE /W4 /permissive- /EHsc)
    target_compile_options(mmb_backend PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(mmb_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(mmb_backend PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(mmb_backend PROPERTIES
    OUTPUT_NAME "mmb_backend"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

if(MMB_WITH_LLAMA)
    add_executable(mmb_llama_acceptance tests/mmb_llama_acceptance.cpp)
    target_link_libraries(mmb_llama_acceptance PRIVATE mmb_core)
    target_compile_features(mmb_llama_acceptance PRIVATE cxx_std_20)
endif()

if(MMB_BUILD_TESTS)
    enable_testing()

    add_executable(mmb_native_tests tests/test_native.cpp)
    target_link_libraries(mmb_native_tests PRIVATE mmb_core mmb_backend)
    target_compile_features(mmb_native_tests PRIVATE cxx_std_20)
    add_test(NAME mmb_native_tests COMMAND mmb_native_tests)

    if(MMB_WITH_LLAMA)
        add_executable(mmb_ggml_tests tests/test_ggml_moe.cpp)
        target_link_libraries(mmb_ggml_tests PRIVATE mmb_core)
        target_compile_features(mmb_ggml_tests PRIVATE cxx_std_20)
        add_test(NAME mmb_ggml_tests COMMAND mmb_ggml_tests)
    endif()
endif()

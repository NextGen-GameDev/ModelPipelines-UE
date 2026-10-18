# G2P bench tools: en_dump (larroy/kokoro.cpp English G2P as-is) and misaki_dump (misaki lexicon port).
# Built by setup_g2p_bench.py; KOKORO_CPP_DIR is a checkout of https://github.com/larroy/kokoro.cpp.
# Only its G2P sources are compiled, so ONNX Runtime is not needed.
cmake_minimum_required(VERSION 3.16)
project(g2p_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT KOKORO_CPP_DIR)
    message(FATAL_ERROR "Set KOKORO_CPP_DIR to a larroy/kokoro.cpp checkout")
endif()

set(LARROY_SRC "${KOKORO_CPP_DIR}/src")
add_library(larroy_g2p STATIC
    "${LARROY_SRC}/english/EnglishPhonemizer.cpp"
    "${LARROY_SRC}/NeuralG2P.cpp"
    "${LARROY_SRC}/NumberNormalizer.cpp")
target_include_directories(larroy_g2p PUBLIC "${LARROY_SRC}" "${LARROY_SRC}/english")

if (MSVC)
    # UTF-8 sources (IPA literals); std::wstring_convert is deprecated but still works.
    add_compile_options(/utf-8 /D_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING)
else()
    add_compile_options(-Wno-deprecated-declarations)
endif()

add_executable(en_dump en_dump.cpp)
target_link_libraries(en_dump PRIVATE larroy_g2p)
add_executable(misaki_dump misaki_dump.cpp)
target_link_libraries(misaki_dump PRIVATE larroy_g2p)

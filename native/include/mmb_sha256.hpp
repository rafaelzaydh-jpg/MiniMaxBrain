#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <filesystem>

namespace mmb {
std::array<uint8_t, 32> sha256_bytes(const uint8_t * data, size_t size);
std::string sha256_hex(const uint8_t * data, size_t size);
std::string sha256_file(const std::filesystem::path & path);
}

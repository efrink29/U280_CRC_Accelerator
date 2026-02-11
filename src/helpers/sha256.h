#ifndef SHA256_H
#define SHA256_H

#include <cstddef>
#include <cstdint>
#include <vector>

std::vector<uint32_t> sha256(const unsigned char *data, size_t len);

#endif // SHA256_H

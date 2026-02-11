#include "sha256.h"

#include <cstring>

namespace
{
static constexpr uint32_t kInit[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

static constexpr uint32_t kRound[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline uint32_t rotr(const uint32_t x, const uint32_t n)
{
    return (x >> n) | (x << (32u - n));
}

inline uint32_t ch(const uint32_t x, const uint32_t y, const uint32_t z)
{
    return (x & y) ^ ((~x) & z);
}

inline uint32_t maj(const uint32_t x, const uint32_t y, const uint32_t z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}

inline uint32_t bsig0(const uint32_t x)
{
    return rotr(x, 2u) ^ rotr(x, 13u) ^ rotr(x, 22u);
}

inline uint32_t bsig1(const uint32_t x)
{
    return rotr(x, 6u) ^ rotr(x, 11u) ^ rotr(x, 25u);
}

inline uint32_t ssig0(const uint32_t x)
{
    return rotr(x, 7u) ^ rotr(x, 18u) ^ (x >> 3u);
}

inline uint32_t ssig1(const uint32_t x)
{
    return rotr(x, 17u) ^ rotr(x, 19u) ^ (x >> 10u);
}

void compress_block(const unsigned char *block, uint32_t state[8])
{
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
    {
        const int base = i << 2;
        w[i] = (static_cast<uint32_t>(block[base]) << 24) |
               (static_cast<uint32_t>(block[base + 1]) << 16) |
               (static_cast<uint32_t>(block[base + 2]) << 8) |
               static_cast<uint32_t>(block[base + 3]);
    }

    for (int i = 16; i < 64; ++i)
    {
        w[i] = ssig1(w[i - 2]) + w[i - 7] + ssig0(w[i - 15]) + w[i - 16];
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (int i = 0; i < 64; ++i)
    {
        const uint32_t t1 = h + bsig1(e) + ch(e, f, g) + kRound[i] + w[i];
        const uint32_t t2 = bsig0(a) + maj(a, b, c);

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}
} // namespace

std::vector<uint32_t> sha256(const unsigned char *data, size_t len)
{
    uint32_t state[8];
    for (int i = 0; i < 8; ++i)
    {
        state[i] = kInit[i];
    }

    const size_t full_blocks = len / 64;
    for (size_t i = 0; i < full_blocks; ++i)
    {
        compress_block(data + i * 64, state);
    }

    const size_t rem = len % 64;
    unsigned char tail[128] = {0};
    if (rem != 0)
    {
        std::memcpy(tail, data + full_blocks * 64, rem);
    }
    tail[rem] = 0x80u;

    const uint64_t bit_len = static_cast<uint64_t>(len) * 8u;
    const size_t len_pos = (rem < 56) ? 56 : 120;
    for (size_t i = 0; i < 8; ++i)
    {
        tail[len_pos + i] = static_cast<unsigned char>((bit_len >> ((7 - i) * 8)) & 0xFFu);
    }

    compress_block(tail, state);
    if (len_pos == 120)
    {
        compress_block(tail + 64, state);
    }

    return std::vector<uint32_t>(state, state + 8);
}

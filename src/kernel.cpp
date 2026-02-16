#include <hls_stream.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <cstdint>
#include <stddef.h>

#define WIDTH 32
#define TABLE_SIZE 256
#define BLOCK_SIZE 16
#define DATA_SIZE 32
#define KERNEL_VARIANT_CRC 1
#define KERNEL_VARIANT_TCP 2
#define KERNEL_VARIANT_SHA 3

#ifndef KERNEL_VARIANT
#define KERNEL_VARIANT KERNEL_VARIANT_CRC
#endif

typedef hls::stream<ap_uint<8>> bStream;

enum KernelCheckMode : uint32_t
{
    CHECK_MODE_CRC = 0,
    CHECK_MODE_TCP_CHECKSUM = 1,
    CHECK_MODE_HASH = 2
};

static const uint32_t SHA256_H[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

static const uint32_t SHA256_K[64] = {
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

static uint32_t rotr32(const uint32_t x, const unsigned int n)
{
#pragma HLS INLINE
    return (x >> n) | (x << (32 - n));
}

static uint32_t sha256_ch(const uint32_t x, const uint32_t y, const uint32_t z)
{
#pragma HLS INLINE
    return (x & y) ^ ((~x) & z);
}

static uint32_t sha256_maj(const uint32_t x, const uint32_t y, const uint32_t z)
{
#pragma HLS INLINE
    return (x & y) ^ (x & z) ^ (y & z);
}

static uint32_t sha256_bsig0(const uint32_t x)
{
#pragma HLS INLINE
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

static uint32_t sha256_bsig1(const uint32_t x)
{
#pragma HLS INLINE
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

static uint32_t sha256_ssig0(const uint32_t x)
{
#pragma HLS INLINE
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

static uint32_t sha256_ssig1(const uint32_t x)
{
#pragma HLS INLINE
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

static void sha256_compress(const unsigned char *block, uint32_t state[8])
{
#pragma HLS INLINE off
    uint32_t w[64];

init_words:
    for (int i = 0; i < 16; ++i)
    {
#pragma HLS PIPELINE II = 1
        const int base = i << 2;
        w[i] = (static_cast<uint32_t>(block[base]) << 24) |
               (static_cast<uint32_t>(block[base + 1]) << 16) |
               (static_cast<uint32_t>(block[base + 2]) << 8) |
               static_cast<uint32_t>(block[base + 3]);
    }

expand_words:
    for (int i = 16; i < 64; ++i)
    {
#pragma HLS PIPELINE II = 1
        w[i] = sha256_ssig1(w[i - 2]) + w[i - 7] + sha256_ssig0(w[i - 15]) + w[i - 16];
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

round_loop:
    for (int i = 0; i < 64; ++i)
    {
#pragma HLS PIPELINE II = 1
        const uint32_t t1 = h + sha256_bsig1(e) + sha256_ch(e, f, g) + SHA256_K[i] + w[i];
        const uint32_t t2 = sha256_bsig0(a) + sha256_maj(a, b, c);

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

static void sha256_process_chunk(const unsigned char *chunk,
                                 const unsigned int chunk_size,
                                 uint32_t digest[8])
{
#pragma HLS INLINE off
    uint32_t state[8];

init_state:
    for (int i = 0; i < 8; ++i)
    {
#pragma HLS UNROLL
        state[i] = SHA256_H[i];
    }

    const unsigned int full_blocks = chunk_size / 64;
    const unsigned int rem = chunk_size % 64;

full_block_loop:
    for (unsigned int b = 0; b < full_blocks; ++b)
    {
        const unsigned char *block = chunk + (static_cast<size_t>(b) << 6);
        sha256_compress(block, state);
    }

    unsigned char pad_block[64];
clear_pad_block:
    for (int i = 0; i < 64; ++i)
    {
#pragma HLS PIPELINE II = 1
        pad_block[i] = 0;
    }

copy_remainder:
    for (unsigned int i = 0; i < rem; ++i)
    {
#pragma HLS PIPELINE II = 1
        pad_block[i] = chunk[(static_cast<size_t>(full_blocks) << 6) + i];
    }
    pad_block[rem] = 0x80;

    const uint64_t bit_len = static_cast<uint64_t>(chunk_size) * 8u;

    if (rem <= 55)
    {
    len_store_single:
        for (int i = 0; i < 8; ++i)
        {
#pragma HLS PIPELINE II = 1
            pad_block[56 + i] = static_cast<unsigned char>((bit_len >> ((7 - i) * 8)) & 0xFF);
        }
        sha256_compress(pad_block, state);
    }
    else
    {
        sha256_compress(pad_block, state);

    clear_second_block:
        for (int i = 0; i < 64; ++i)
        {
#pragma HLS PIPELINE II = 1
            pad_block[i] = 0;
        }

    len_store_double:
        for (int i = 0; i < 8; ++i)
        {
#pragma HLS PIPELINE II = 1
            pad_block[56 + i] = static_cast<unsigned char>((bit_len >> ((7 - i) * 8)) & 0xFF);
        }
        sha256_compress(pad_block, state);
    }

write_digest:
    for (int i = 0; i < 8; ++i)
    {
#pragma HLS UNROLL
        digest[i] = state[i];
    }
}

static void read_input(const unsigned char *in,
                       bStream &b0, bStream &b1, bStream &b2, bStream &b3,
                       bStream &b4, bStream &b5, bStream &b6, bStream &b7,
                       bStream &b8, bStream &b9, bStream &b10, bStream &b11,
                       bStream &b12, bStream &b13, bStream &b14, bStream &b15,
                       const unsigned int numChunks, const unsigned int chunkSize)
{
    const int total_bytes = numChunks * (int)chunkSize;
    const int loop_count = total_bytes / 16;
    const ap_uint<128> *p128 = reinterpret_cast<const ap_uint<128> *>(in);

mem_rd:
    for (int i = 0; i < loop_count; ++i)
    {
#pragma HLS PIPELINE II = 1
        ap_uint<128> data = p128[i];
        b0 << data(7, 0);
        b1 << data(15, 8);
        b2 << data(23, 16);
        b3 << data(31, 24);
        b4 << data(39, 32);
        b5 << data(47, 40);
        b6 << data(55, 48);
        b7 << data(63, 56);
        b8 << data(71, 64);
        b9 << data(79, 72);
        b10 << data(87, 80);
        b11 << data(95, 88);
        b12 << data(103, 96);
        b13 << data(111, 104);
        b14 << data(119, 112);
        b15 << data(127, 120);
    }
}

static void process_blocks(
    uint32_t crcTables[BLOCK_SIZE][TABLE_SIZE],
    bStream &byte0, bStream &byte1, bStream &byte2, bStream &byte3,
    bStream &byte4, bStream &byte5, bStream &byte6, bStream &byte7,
    bStream &byte8, bStream &byte9, bStream &byte10, bStream &byte11,
    bStream &byte12, bStream &byte13, bStream &byte14, bStream &byte15,
    const uint32_t crc_size, const uint32_t init_value,
    hls::stream<ap_uint<32>> &outStream,
    int numChunks, int chunkSize)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = crcTables complete dim = 1
#pragma HLS BIND_STORAGE variable = crcTables type = ram_1p impl = bram

    ap_uint<32> mask = 0xFFFFFFFFu;
    if (crc_size < 32)
        mask = (1u << crc_size) - 1u;

chunk_loop:
    for (int c = 0; c < numChunks; ++c)
    {
        ap_uint<32> crc = init_value;

        const int blocks_in_chunk = chunkSize / 16;
        const int tail_bytes = chunkSize - blocks_in_chunk * 16;

    block_loop:
        for (int b = 0; b < blocks_in_chunk; ++b)
        {
#pragma HLS PIPELINE II = 1

            ap_uint<8> b0 = byte0.read();
            ap_uint<8> b1 = byte1.read();
            ap_uint<8> b2 = byte2.read();
            ap_uint<8> b3 = byte3.read();
            ap_uint<8> b4 = byte4.read();
            ap_uint<8> b5 = byte5.read();
            ap_uint<8> b6 = byte6.read();
            ap_uint<8> b7 = byte7.read();
            ap_uint<8> b8 = byte8.read();
            ap_uint<8> b9 = byte9.read();
            ap_uint<8> b10 = byte10.read();
            ap_uint<8> b11 = byte11.read();
            ap_uint<8> b12 = byte12.read();
            ap_uint<8> b13 = byte13.read();
            ap_uint<8> b14 = byte14.read();
            ap_uint<8> b15 = byte15.read();

            ap_uint<32> next =
                (crcTables[0][((crc) & 0xFF) ^ b0] ^
                 crcTables[1][((crc >> 8) & 0xFF) ^ b1]) ^
                (crcTables[2][((crc >> 16) & 0xFF) ^ b2] ^
                 crcTables[3][((crc >> 24) & 0xFF) ^ b3]) ^
                (crcTables[4][b4] ^ crcTables[5][b5]) ^
                (crcTables[6][b6] ^ crcTables[7][b7]) ^
                (crcTables[8][b8] ^ crcTables[9][b9]) ^
                (crcTables[10][b10] ^ crcTables[11][b11]) ^
                (crcTables[12][b12] ^ crcTables[13][b13]) ^
                (crcTables[14][b14] ^ crcTables[15][b15]);

            crc = next & mask;
        }

        if (tail_bytes)
        {
        tail_loop:
            for (int t = 0; t < tail_bytes; ++t)
            {
#pragma HLS PIPELINE II = 1
                ap_uint<8> bt;
                switch (t & 15)
                {
                case 0:
                    bt = byte0.read();
                    break;
                case 1:
                    bt = byte1.read();
                    break;
                case 2:
                    bt = byte2.read();
                    break;
                case 3:
                    bt = byte3.read();
                    break;
                case 4:
                    bt = byte4.read();
                    break;
                case 5:
                    bt = byte5.read();
                    break;
                case 6:
                    bt = byte6.read();
                    break;
                case 7:
                    bt = byte7.read();
                    break;
                case 8:
                    bt = byte8.read();
                    break;
                case 9:
                    bt = byte9.read();
                    break;
                case 10:
                    bt = byte10.read();
                    break;
                case 11:
                    bt = byte11.read();
                    break;
                case 12:
                    bt = byte12.read();
                    break;
                case 13:
                    bt = byte13.read();
                    break;
                case 14:
                    bt = byte14.read();
                    break;
                default:
                    bt = byte15.read();
                    break;
                }

                crc = ((crc >> 8) ^ crcTables[0][(crc ^ bt) & 0xFF]) & mask;
            }
        }

        outStream << (ap_uint<32>)crc;
    }
}

static void write_output(hls::stream<ap_uint<32>> &outStream, int numChunks, uint32_t *crc_out)
{
    for (int i = 0; i < numChunks; i++)
    {
#pragma HLS PIPELINE II = 1
        crc_out[i] = static_cast<uint32_t>(outStream.read());
    }
}

static ap_uint<17> fold_add(ap_uint<17> sum, ap_uint<16> word)
{
#pragma HLS INLINE
    ap_uint<17> tmp = sum + word;
    return (tmp & 0xFFFF) + (tmp >> 16);
}

static ap_uint<16> tcp_checksum_chunk(const unsigned char *chunk, const unsigned int chunk_size)
{
#pragma HLS INLINE
    ap_uint<17> sum = 0;
    unsigned int i = 0;

word_loop:
    for (; i + 1 < chunk_size; i += 2)
    {
#pragma HLS PIPELINE II = 1
        const ap_uint<16> word =
            (static_cast<ap_uint<16>>(chunk[i]) << 8) |
            static_cast<ap_uint<16>>(chunk[i + 1]);
        sum = fold_add(sum, word);
    }

    if (i < chunk_size)
    {
        const ap_uint<16> last_word = static_cast<ap_uint<16>>(chunk[i]) << 8;
        sum = fold_add(sum, last_word);
    }

    ap_uint<16> folded = (sum & 0xFFFF) + (sum >> 16);
    folded = (folded & 0xFFFF) + (folded >> 16);
    return ~folded;
}

static void process_tcp_checksum(const unsigned char *data_in,
                                 uint32_t *crc_out,
                                 const unsigned int numChunks,
                                 const unsigned int chunkSize)
{
tcp_chunk_loop:
    for (unsigned int c = 0; c < numChunks; ++c)
    {
#pragma HLS PIPELINE II = 1
        const unsigned char *chunk = data_in + (static_cast<size_t>(c) * chunkSize);
        const ap_uint<16> checksum = tcp_checksum_chunk(chunk, chunkSize);
        crc_out[c] = static_cast<uint32_t>(checksum);
    }
}

static void process_hash(const unsigned char *data_in,
                         uint32_t *crc_out,
                         const unsigned int numChunks,
                         const unsigned int chunkSize)
{
hash_chunk_loop:
    for (unsigned int c = 0; c < numChunks; ++c)
    {
        const unsigned char *chunk = data_in + (static_cast<size_t>(c) * chunkSize);
        uint32_t digest[8];
#pragma HLS ARRAY_PARTITION variable = digest complete dim = 1
        sha256_process_chunk(chunk, chunkSize, digest);

    hash_write_digest:
        for (int i = 0; i < 8; ++i)
        {
#pragma HLS PIPELINE II = 1
            crc_out[(static_cast<size_t>(c) << 3) + i] = digest[i];
        }
    }
}

static void process_crc(const unsigned char *data_in,
                        uint32_t *crc_out,
                        const uint32_t *tables,
                        const unsigned int numChunks,
                        const unsigned int chunkSize,
                        const uint32_t crc_size,
                        const uint32_t init_value)
{
    bStream inByte0;
    bStream inByte1;
    bStream inByte2;
    bStream inByte3;

    bStream inByte4;
    bStream inByte5;
    bStream inByte6;
    bStream inByte7;

    bStream inByte8;
    bStream inByte9;
    bStream inByte10;
    bStream inByte11;

    bStream inByte12;
    bStream inByte13;
    bStream inByte14;
    bStream inByte15;
#pragma HLS STREAM variable = inByte0 depth = 64
#pragma HLS STREAM variable = inByte1 depth = 64
#pragma HLS STREAM variable = inByte2 depth = 64
#pragma HLS STREAM variable = inByte3 depth = 64
#pragma HLS STREAM variable = inByte4 depth = 64
#pragma HLS STREAM variable = inByte5 depth = 64
#pragma HLS STREAM variable = inByte6 depth = 64
#pragma HLS STREAM variable = inByte7 depth = 64
#pragma HLS STREAM variable = inByte8 depth = 64
#pragma HLS STREAM variable = inByte9 depth = 64
#pragma HLS STREAM variable = inByte10 depth = 64
#pragma HLS STREAM variable = inByte11 depth = 64
#pragma HLS STREAM variable = inByte12 depth = 64
#pragma HLS STREAM variable = inByte13 depth = 64
#pragma HLS STREAM variable = inByte14 depth = 64
#pragma HLS STREAM variable = inByte15 depth = 64

    hls::stream<ap_uint<32>> outStream;
#pragma HLS STREAM variable = outStream depth = 64

    uint32_t crcTables[BLOCK_SIZE][TABLE_SIZE];
#pragma HLS ARRAY_PARTITION variable = crcTables complete dim = 1
#pragma HLS BIND_STORAGE variable = crcTables type = ram_1p impl = bram

init_lut:
    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
        for (int j = 0; j < TABLE_SIZE; ++j)
        {
#pragma HLS PIPELINE II = 1
            crcTables[i][j] = tables[(i << 8) + j];
        }
    }

#pragma HLS DATAFLOW
    read_input(data_in, inByte0, inByte1, inByte2, inByte3, inByte4, inByte5, inByte6, inByte7,
               inByte8, inByte9, inByte10, inByte11, inByte12, inByte13, inByte14, inByte15,
               numChunks, chunkSize);
    process_blocks(crcTables, inByte0, inByte1, inByte2, inByte3, inByte4, inByte5, inByte6, inByte7,
                   inByte8, inByte9, inByte10, inByte11, inByte12, inByte13, inByte14, inByte15,
                   crc_size, init_value, outStream, numChunks, chunkSize);
    write_output(outStream, numChunks, crc_out);
}

extern "C"
{
#if KERNEL_VARIANT == KERNEL_VARIANT_CRC
    void calculate_crc(const unsigned char *data_in,
                       uint32_t *crc_out,
                       const uint32_t *tables,
                       const unsigned int numChunks,
                       const unsigned int chunkSize,
                       const uint32_t crc_size,
                       const uint32_t init_value)
    {
#pragma HLS INTERFACE m_axi port = data_in bundle = gmem0 depth = 1024 offset = slave
#pragma HLS INTERFACE m_axi port = tables bundle = gmem1 depth = 4096 offset = slave
#pragma HLS INTERFACE m_axi port = crc_out bundle = gmem0 depth = 1024 offset = slave
        process_crc(data_in, crc_out, tables, numChunks, chunkSize, crc_size, init_value);
    }
#endif

#if KERNEL_VARIANT == KERNEL_VARIANT_TCP
    void calculate_tcp_checksum(const unsigned char *data_in,
                                uint32_t *crc_out,
                                const unsigned int numChunks,
                                const unsigned int chunkSize)
    {
#pragma HLS INTERFACE m_axi port = data_in bundle = gmem0 depth = 1024 offset = slave
#pragma HLS INTERFACE m_axi port = crc_out bundle = gmem0 depth = 1024 offset = slave
        process_tcp_checksum(data_in, crc_out, numChunks, chunkSize);
    }
#endif

#if KERNEL_VARIANT == KERNEL_VARIANT_SHA
    void calculate_sha256(const unsigned char *data_in,
                          uint32_t *crc_out,
                          const unsigned int numChunks,
                          const unsigned int chunkSize)
    {
#pragma HLS INTERFACE m_axi port = data_in bundle = gmem0 depth = 1024 offset = slave
#pragma HLS INTERFACE m_axi port = crc_out bundle = gmem0 depth = 1024 offset = slave
        process_hash(data_in, crc_out, numChunks, chunkSize);
    }
#endif
}

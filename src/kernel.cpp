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
typedef ap_uint<512> wide_t;

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

// ------------------ SHA-256 Kernel Implementation ----------------------
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

static ap_uint<16> to_be16(const ap_uint<16> x)
{
#pragma HLS INLINE
    return (x << 8) | (x >> 8);
}

static uint32_t to_be32(const ap_uint<32> x)
{
#pragma HLS INLINE
    return (static_cast<uint32_t>(x.range(7, 0)) << 24) |
           (static_cast<uint32_t>(x.range(15, 8)) << 16) |
           (static_cast<uint32_t>(x.range(23, 16)) << 8) |
           static_cast<uint32_t>(x.range(31, 24));
}

static wide_t pack_block_64(const unsigned char *block)
{
#pragma HLS INLINE
    wide_t packed = 0;
pack_block_bytes:
    for (int i = 0; i < 64; ++i)
    {
#pragma HLS UNROLL
        packed.range((i * 8) + 7, i * 8) = block[i];
    }
    return packed;
}

static void sha256_compress_block(const wide_t &block, uint32_t state[8])
{
#pragma HLS INLINE off

    uint32_t wbuf[16];
#pragma HLS ARRAY_PARTITION variable = wbuf complete dim = 1

init_words:
    for (int i = 0; i < 16; ++i)
    {
#pragma HLS UNROLL
        const ap_uint<32> word_le = block.range((i * 32) + 31, i * 32);
        wbuf[i] = to_be32(word_le);
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
        uint32_t wt;

        if (i < 16)
        {
            wt = wbuf[i];
        }
        else
        {
            const int idx = i & 15;
            const uint32_t s0 = sha256_ssig0(wbuf[(i - 15) & 15]);
            const uint32_t s1 = sha256_ssig1(wbuf[(i - 2) & 15]);
            wt = s1 + wbuf[(i - 7) & 15] + s0 + wbuf[(i - 16) & 15];
            wbuf[idx] = wt;
        }

        const uint32_t t1 = h + sha256_bsig1(e) + sha256_ch(e, f, g) + SHA256_K[i] + wt;
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
                                 uint32_t digest[8]);

static void sha256_process_and_write_chunk(const unsigned char *chunk,
                                           const unsigned int chunk_size,
                                           uint32_t *out_words)
{
#pragma HLS INLINE off

    uint32_t digest[8];
#pragma HLS ARRAY_PARTITION variable = digest complete dim = 1

    sha256_process_chunk(chunk, chunk_size, digest);

write_digest:
    for (int i = 0; i < 8; ++i)
    {
#pragma HLS PIPELINE II = 1
        out_words[i] = digest[i];
    }
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
    const bool vector_blocks = ((chunk_size & 63u) == 0u);

    if (vector_blocks)
    {
        const wide_t *chunk_words = reinterpret_cast<const wide_t *>(chunk);
    full_block_loop_vec:
        for (unsigned int b = 0; b < full_blocks; ++b)
        {
            const wide_t block = chunk_words[b];
            sha256_compress_block(block, state);
        }
    }
    else
    {
    full_block_loop_scalar:
        for (unsigned int b = 0; b < full_blocks; ++b)
        {
            const unsigned char *block = chunk + (static_cast<size_t>(b) << 6);
            const wide_t packed = pack_block_64(block);
            sha256_compress_block(packed, state);
        }
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
        const wide_t packed_pad = pack_block_64(pad_block);
        sha256_compress_block(packed_pad, state);
    }
    else
    {
        wide_t packed_pad = pack_block_64(pad_block);
        sha256_compress_block(packed_pad, state);

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
        packed_pad = pack_block_64(pad_block);
        sha256_compress_block(packed_pad, state);
    }

write_digest:
    for (int i = 0; i < 8; ++i)
    {
#pragma HLS UNROLL
        digest[i] = state[i];
    }
}

// ------------------ CRC Kernel Implementation ----------------------

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

static void write_output(hls::stream<ap_uint<32>> &outStream,
                         int numChunks,
                         uint32_t *crc_out)
{
    for (int i = 0; i < numChunks; i++)
    {
#pragma HLS PIPELINE II = 1
        crc_out[i] = static_cast<uint32_t>(outStream.read());
    }
}

static void load_crc_tables(const uint32_t *tables,
                            uint32_t crcTables[BLOCK_SIZE][TABLE_SIZE])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = crcTables complete dim = 1
#pragma HLS BIND_STORAGE variable = crcTables type = ram_1p impl = bram

load_lut_rows:
    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
    load_lut_cols:
        for (int j = 0; j < TABLE_SIZE; ++j)
        {
#pragma HLS PIPELINE II = 1
            crcTables[i][j] = tables[(i << 8) + j];
        }
    }
}

static ap_uint<32> crc_process_full_blocks(
    uint32_t crcTables[BLOCK_SIZE][TABLE_SIZE],
    bStream &byte0, bStream &byte1, bStream &byte2, bStream &byte3,
    bStream &byte4, bStream &byte5, bStream &byte6, bStream &byte7,
    bStream &byte8, bStream &byte9, bStream &byte10, bStream &byte11,
    bStream &byte12, bStream &byte13, bStream &byte14, bStream &byte15,
    const int blocks_in_chunk,
    ap_uint<32> crc,
    const ap_uint<32> mask)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = crcTables complete dim = 1
#pragma HLS BIND_STORAGE variable = crcTables type = ram_1p impl = bram

full_block_loop:
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

    return crc;
}

static ap_uint<32> crc_process_tail_bytes(
    uint32_t crcTables[BLOCK_SIZE][TABLE_SIZE],
    bStream &byte0, bStream &byte1, bStream &byte2, bStream &byte3,
    bStream &byte4, bStream &byte5, bStream &byte6, bStream &byte7,
    bStream &byte8, bStream &byte9, bStream &byte10, bStream &byte11,
    bStream &byte12, bStream &byte13, bStream &byte14, bStream &byte15,
    const int tail_bytes,
    ap_uint<32> crc,
    const ap_uint<32> mask)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = crcTables complete dim = 1
#pragma HLS BIND_STORAGE variable = crcTables type = ram_1p impl = bram

tail_byte_loop:
    for (int t = 0; t < tail_bytes; ++t)
    {
#pragma HLS PIPELINE II = 1
        ap_uint<8> bt = 0;

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

    return crc;
}

static void process_crc_chunks(
    uint32_t crcTables[BLOCK_SIZE][TABLE_SIZE],
    bStream &byte0, bStream &byte1, bStream &byte2, bStream &byte3,
    bStream &byte4, bStream &byte5, bStream &byte6, bStream &byte7,
    bStream &byte8, bStream &byte9, bStream &byte10, bStream &byte11,
    bStream &byte12, bStream &byte13, bStream &byte14, bStream &byte15,
    const uint32_t crc_size,
    const uint32_t init_value,
    hls::stream<ap_uint<32>> &outStream,
    const int numChunks,
    const int chunkSize)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = crcTables complete dim = 1
#pragma HLS BIND_STORAGE variable = crcTables type = ram_1p impl = bram

    ap_uint<32> mask = 0xFFFFFFFFu;
    if (crc_size < 32)
        mask = (1u << crc_size) - 1u;

    const int blocks_in_chunk = chunkSize / 16;
    const int tail_bytes = chunkSize - (blocks_in_chunk * 16);

chunk_loop:
    for (int c = 0; c < numChunks; ++c)
    {
        ap_uint<32> crc = init_value;

        crc = crc_process_full_blocks(
            crcTables,
            byte0, byte1, byte2, byte3,
            byte4, byte5, byte6, byte7,
            byte8, byte9, byte10, byte11,
            byte12, byte13, byte14, byte15,
            blocks_in_chunk, crc, mask);

        if (tail_bytes > 0)
        {
            crc = crc_process_tail_bytes(
                crcTables,
                byte0, byte1, byte2, byte3,
                byte4, byte5, byte6, byte7,
                byte8, byte9, byte10, byte11,
                byte12, byte13, byte14, byte15,
                tail_bytes, crc, mask);
        }

        outStream << crc;
    }
}

static void crc_dataflow_region(const unsigned char *data_in,
                                uint32_t *crc_out,
                                uint32_t crcTables[BLOCK_SIZE][TABLE_SIZE],
                                const unsigned int numChunks,
                                const unsigned int chunkSize,
                                const uint32_t crc_size,
                                const uint32_t init_value)
{
#pragma HLS INLINE off

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

#pragma HLS DATAFLOW
    read_input(data_in,
               inByte0, inByte1, inByte2, inByte3,
               inByte4, inByte5, inByte6, inByte7,
               inByte8, inByte9, inByte10, inByte11,
               inByte12, inByte13, inByte14, inByte15,
               numChunks, chunkSize);

    process_crc_chunks(crcTables,
                       inByte0, inByte1, inByte2, inByte3,
                       inByte4, inByte5, inByte6, inByte7,
                       inByte8, inByte9, inByte10, inByte11,
                       inByte12, inByte13, inByte14, inByte15,
                       crc_size, init_value, outStream,
                       static_cast<int>(numChunks),
                       static_cast<int>(chunkSize));

    write_output(outStream, static_cast<int>(numChunks), crc_out);
}

// ------------------ TCP Checksum Kernel Implementation ----------------------

static ap_uint<17> fold_add(ap_uint<17> sum, ap_uint<16> word)
{
#pragma HLS INLINE
    ap_uint<17> tmp = sum + word;
    return (tmp & 0xFFFF) + (tmp >> 16);
}

static ap_uint<16> tcp_checksum_chunk_scalar(const unsigned char *chunk, const unsigned int chunk_size)
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

static ap_uint<16> tcp_checksum_chunk_vector(const wide_t *chunk_words,
                                             const unsigned char *chunk_bytes,
                                             const unsigned int chunk_size)
{
#pragma HLS INLINE
    ap_uint<48> sum = 0;
    const unsigned int vec_words = chunk_size >> 6;

vec_word_loop:
    for (unsigned int w = 0; w < vec_words; ++w)
    {
#pragma HLS PIPELINE II = 1
        const wide_t packed = chunk_words[w];

        ap_uint<17> lvl1[16];
        ap_uint<18> lvl2[8];
        ap_uint<19> lvl3[4];
        ap_uint<20> lvl4[2];
#pragma HLS ARRAY_PARTITION variable = lvl1 complete
#pragma HLS ARRAY_PARTITION variable = lvl2 complete
#pragma HLS ARRAY_PARTITION variable = lvl3 complete
#pragma HLS ARRAY_PARTITION variable = lvl4 complete

    reduce_lvl1:
        for (int i = 0; i < 16; ++i)
        {
#pragma HLS UNROLL
            const int idx0 = i << 1;
            const int idx1 = idx0 + 1;
            const ap_uint<16> word0 = to_be16(packed.range((idx0 * 16) + 15, idx0 * 16));
            const ap_uint<16> word1 = to_be16(packed.range((idx1 * 16) + 15, idx1 * 16));
            lvl1[i] = static_cast<ap_uint<17>>(word0) + static_cast<ap_uint<17>>(word1);
        }

    reduce_lvl2:
        for (int i = 0; i < 8; ++i)
        {
#pragma HLS UNROLL
            lvl2[i] = static_cast<ap_uint<18>>(lvl1[i << 1]) + static_cast<ap_uint<18>>(lvl1[(i << 1) + 1]);
        }

    reduce_lvl3:
        for (int i = 0; i < 4; ++i)
        {
#pragma HLS UNROLL
            lvl3[i] = static_cast<ap_uint<19>>(lvl2[i << 1]) + static_cast<ap_uint<19>>(lvl2[(i << 1) + 1]);
        }

    reduce_lvl4:
        for (int i = 0; i < 2; ++i)
        {
#pragma HLS UNROLL
            lvl4[i] = static_cast<ap_uint<20>>(lvl3[i << 1]) + static_cast<ap_uint<20>>(lvl3[(i << 1) + 1]);
        }

        sum += static_cast<ap_uint<48>>(lvl4[0]) + static_cast<ap_uint<48>>(lvl4[1]);
    }

    const unsigned int tail_offset = vec_words << 6;
    const unsigned int tail_bytes = chunk_size - tail_offset;

tail_word_loop:
    for (unsigned int i = 0; i + 1 < tail_bytes; i += 2)
    {
#pragma HLS PIPELINE II = 1
        const ap_uint<16> word =
            (static_cast<ap_uint<16>>(chunk_bytes[tail_offset + i]) << 8) |
            static_cast<ap_uint<16>>(chunk_bytes[tail_offset + i + 1]);
        sum += word;
    }

    if ((tail_bytes & 1u) != 0u)
    {
        const ap_uint<16> last_word =
            static_cast<ap_uint<16>>(chunk_bytes[tail_offset + tail_bytes - 1]) << 8;
        sum += last_word;
    }

    ap_uint<32> folded = static_cast<ap_uint<32>>(sum);
fold_carry:
    for (int i = 0; i < 4; ++i)
    {
#pragma HLS UNROLL
        folded = (folded & 0xFFFF) + (folded >> 16);
    }
    return static_cast<ap_uint<16>>(~folded);
}

static void process_tcp_checksum(const wide_t *data_in,
                                 uint32_t *crc_out,
                                 const unsigned int numChunks,
                                 const unsigned int chunkSize)
{
    const unsigned char *data_bytes = reinterpret_cast<const unsigned char *>(data_in);
    const bool vector_blocks = ((chunkSize & 63u) == 0u);

tcp_chunk_loop:
    for (unsigned int c = 0; c < numChunks; ++c)
    {
#pragma HLS PIPELINE II = 1
        const size_t chunk_offset = static_cast<size_t>(c) * chunkSize;
        const unsigned char *chunk_bytes = data_bytes + chunk_offset;

        ap_uint<16> checksum = 0;
        if (vector_blocks)
        {
            const wide_t *chunk_words = data_in + (chunk_offset >> 6);
            checksum = tcp_checksum_chunk_vector(chunk_words, chunk_bytes, chunkSize);
        }
        else
        {
            checksum = tcp_checksum_chunk_scalar(chunk_bytes, chunkSize);
        }
        crc_out[c] = static_cast<uint32_t>(checksum);
    }
}

static void process_hash(const wide_t *data_in,
                         uint32_t *crc_out,
                         const unsigned int numChunks,
                         const unsigned int chunkSize)
{
    const unsigned char *data_bytes = reinterpret_cast<const unsigned char *>(data_in);

hash_chunk_loop:
    for (unsigned int c = 0; c < numChunks; ++c)
    {
        const unsigned char *chunk = data_bytes + (static_cast<size_t>(c) * chunkSize);
        uint32_t *digest_out = crc_out + (static_cast<size_t>(c) << 3);
        sha256_process_and_write_chunk(chunk, chunkSize, digest_out);
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
    uint32_t crcTables[BLOCK_SIZE][TABLE_SIZE];
#pragma HLS ARRAY_PARTITION variable = crcTables complete dim = 1
#pragma HLS BIND_STORAGE variable = crcTables type = ram_1p impl = bram

    load_crc_tables(tables, crcTables);

    crc_dataflow_region(data_in, crc_out, crcTables,
                        numChunks, chunkSize, crc_size, init_value);
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
    void calculate_tcp_checksum(const wide_t *data_in,
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
    void calculate_sha256(const wide_t *data_in,
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
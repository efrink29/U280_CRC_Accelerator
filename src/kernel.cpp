

#include <hls_stream.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <cstdint>
#include <stddef.h>

#define WIDTH 32
#define TABLE_SIZE 256
#define BLOCK_SIZE 16
#define DATA_SIZE 32

typedef hls::stream<ap_uint<8>> bStream;

void read_input(const unsigned char *in,
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

    // guard mask
    ap_uint<32> mask = 0xFFFFFFFFu;
    if (crc_size < 32)
        mask = (1u << crc_size) - 1u;

    // enforce chunk structure to HLS (prevents boundary reordering)
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

        // If your producer can emit non-multiple-of-16 tails per chunk, handle them here.
        // (If you never have tails, you can remove this section.)
        if (tail_bytes)
        {
        tail_loop:
            for (int t = 0; t < tail_bytes; ++t)
            {
#pragma HLS PIPELINE II = 1
                // Read only the lane that corresponds to position t (0..15)
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
    ///*
    for (int i = 0; i < numChunks; i++)
    {
#pragma HLS PIPELINE II = 1 //*/
        crc_out[i] = static_cast<uint32_t>(outStream.read());
    }
}

extern "C"
{
    void calculate_crc(const unsigned char *data_in, uint32_t *crc_out, const uint32_t *tables, const unsigned int numChunks, const unsigned int chunkSize, const uint32_t crc_size, const uint32_t init_value)
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
        // repeat for all 16 input streams
        hls::stream<ap_uint<32>>
            outStream;
#pragma HLS STREAM variable = outStream depth = 64

        uint32_t crcTables[BLOCK_SIZE][TABLE_SIZE];
#pragma HLS ARRAY_PARTITION variable = crcTables complete dim = 1
#pragma HLS BIND_STORAGE variable = crcTables type = ram_1p impl = bram

    // Load once from AXI
    init_lut:
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < TABLE_SIZE; ++j)
            {
#pragma HLS PIPELINE II = 1
                crcTables[i][j] = tables[(i << 8) + j];
            }
        }
#pragma HLS INTERFACE m_axi port = data_in bundle = gmem0 depth = 1024 offset = slave
#pragma HLS INTERFACE m_axi port = tables bundle = gmem1 depth = 4096 offset = slave
#pragma HLS INTERFACE m_axi port = crc_out bundle = gmem0 depth = 1024 offset = slave

#pragma HLS DATAFLOW
        read_input(data_in, inByte0, inByte1, inByte2, inByte3, inByte4, inByte5, inByte6, inByte7, inByte8, inByte9, inByte10, inByte11, inByte12, inByte13, inByte14, inByte15, numChunks, chunkSize);
        process_blocks(crcTables, inByte0, inByte1, inByte2, inByte3, inByte4, inByte5, inByte6, inByte7, inByte8, inByte9, inByte10, inByte11, inByte12, inByte13, inByte14, inByte15, crc_size, init_value, outStream, numChunks, chunkSize);
        write_output(outStream, numChunks, crc_out);
    }
}

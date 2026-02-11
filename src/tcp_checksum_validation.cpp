#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
#include "helpers/crc.h"

extern "C" void calculate_crc(const unsigned char *data_in,
                               uint32_t *crc_out,
                               const uint32_t *tables,
                               unsigned int numChunks,
                               unsigned int chunkSize,
                               uint32_t crc_size,
                               uint32_t init_value,
                               uint32_t mode);

static uint16_t mled_tcp_checksum(const unsigned char *data, size_t len)
{
    unsigned int sum = 0;
    size_t i = 0;

    for (; i + 1 < len; i += 2)
    {
        sum += (static_cast<unsigned int>(static_cast<unsigned char>(data[i])) << 8) |
               static_cast<unsigned int>(static_cast<unsigned char>(data[i + 1]));
        while (sum >> 16)
        {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
    }

    if (i < len)
    {
        sum += static_cast<unsigned int>(static_cast<unsigned char>(data[i])) << 8;
        while (sum >> 16)
        {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
    }

    sum = ~sum;
    return static_cast<uint16_t>(sum & 0xFFFF);
}

int main()
{
    constexpr uint32_t CHECK_MODE_CRC = 0;
    constexpr uint32_t CHECK_MODE_TCP_CHECKSUM = 1;

    {
        const unsigned char rfc_data[] = {0x00, 0x01, 0xF2, 0x03, 0xF4, 0xF5, 0xF6, 0xF7};
        const uint16_t checksum = mled_tcp_checksum(rfc_data, sizeof(rfc_data));
        if (checksum != 0x220D)
        {
            std::cerr << "RFC test vector failed. Got 0x" << std::hex << checksum
                      << ", expected 0x220D" << std::dec << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::vector<unsigned int> chunk_sizes = {1, 2, 3, 7, 15, 16, 31, 64, 255, 1024};
    std::vector<uint32_t> dummy_tables(16 * 256, 0);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> byte_dist(0, 255);

    for (unsigned int chunk_size : chunk_sizes)
    {
        for (unsigned int num_chunks = 1; num_chunks <= 8; ++num_chunks)
        {
            std::vector<unsigned char> data(static_cast<size_t>(chunk_size) * num_chunks);
            for (size_t i = 0; i < data.size(); ++i)
            {
                data[i] = static_cast<unsigned char>(byte_dist(rng));
            }

            std::vector<uint32_t> fpga_out(num_chunks, 0);
            calculate_crc(data.data(),
                          fpga_out.data(),
                          dummy_tables.data(),
                          num_chunks,
                          chunk_size,
                          16,
                          0,
                          CHECK_MODE_TCP_CHECKSUM);

            for (unsigned int c = 0; c < num_chunks; ++c)
            {
                const uint16_t expected = mled_tcp_checksum(data.data() + (static_cast<size_t>(c) * chunk_size),
                                                            chunk_size);
                const uint16_t actual = static_cast<uint16_t>(fpga_out[c] & 0xFFFF);
                if (actual != expected)
                {
                    std::cerr << "Mismatch at chunk_size=" << chunk_size
                              << ", num_chunks=" << num_chunks
                              << ", chunk_index=" << c
                              << ". got=0x" << std::hex << actual
                              << " expected=0x" << expected << std::dec << std::endl;
                    return EXIT_FAILURE;
                }
            }
        }
    }

    {
        CRC_Config cfg{};
        cfg.polynomial = 0x04C11DB7;
        cfg.initial_value = 0xFFFFFFFF;
        cfg.reflect_input = true;
        cfg.reflect_output = true;
        cfg.final_xor_value = 0xFFFFFFFF;
        cfg.width = 32;
        cfg.chunk_size = 64;

        const unsigned int chunk_size = static_cast<unsigned int>(cfg.chunk_size);
        const unsigned int num_chunks = 4;

        auto tables = create_parallel_tables(cfg);
        std::vector<uint32_t> flatTbl(16 * 256, 0);
        for (int i = 0; i < 16; ++i)
        {
            for (int j = 0; j < 256; ++j)
            {
                flatTbl[(i << 8) + j] = tables[i][j];
            }
        }

        std::mt19937 rng_crc(4242);
        std::uniform_int_distribution<int> byte_dist_crc(0, 255);
        const size_t total_bytes = static_cast<size_t>(chunk_size) * num_chunks;
        void *aligned_mem = nullptr;
        if (posix_memalign(&aligned_mem, 16, total_bytes) != 0 || aligned_mem == nullptr)
        {
            std::cerr << "Failed to allocate aligned test buffer for CRC validation." << std::endl;
            return EXIT_FAILURE;
        }
        unsigned char *data = static_cast<unsigned char *>(aligned_mem);
        for (size_t i = 0; i < total_bytes; ++i)
        {
            data[i] = static_cast<unsigned char>(byte_dist_crc(rng_crc));
        }

        std::vector<uint32_t> fpga_out(num_chunks, 0);
        calculate_crc(data,
                      fpga_out.data(),
                      flatTbl.data(),
                      num_chunks,
                      chunk_size,
                      cfg.width,
                      cfg.initial_value,
                      CHECK_MODE_CRC);

        for (unsigned int c = 0; c < num_chunks; ++c)
        {
            uint32_t expected = parallel_compute(data + (static_cast<size_t>(c) * chunk_size),
                                                 chunk_size,
                                                 cfg);
            uint32_t actual = fpga_out[c] ^ cfg.final_xor_value;
            if (cfg.reflect_output != cfg.reflect_input)
            {
                actual = reflect(actual, cfg.width);
            }
            if (actual != expected)
            {
                std::cerr << "CRC mismatch at chunk " << c
                          << ". got=0x" << std::hex << actual
                          << " expected=0x" << expected << std::dec << std::endl;
                return EXIT_FAILURE;
            }
        }
        free(aligned_mem);
    }

    std::cout << "PASS: kernel dual-mode matches software references (TCP + CRC)." << std::endl;
    return EXIT_SUCCESS;
}

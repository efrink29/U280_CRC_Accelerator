#ifndef CRC_H
#define CRC_H

#include <cstdint>
#include <cstddef>
#include <iostream>

struct CRC_Config
{
    uint32_t polynomial;
    uint32_t initial_value;
    bool reflect_input;
    bool reflect_output;
    uint32_t final_xor_value;
    uint8_t width;
    size_t chunk_size;
};

uint32_t *create_standard_table(const CRC_Config &config);
uint32_t **create_parallel_tables(const CRC_Config &config);

uint32_t standard_compute(const uint8_t *data, size_t length, const CRC_Config &config, bool table_gen = false);

uint32_t standard_compute_golden(const uint8_t *data, size_t length, const CRC_Config &config);

uint32_t parallel_compute(const uint8_t *data, size_t length, const CRC_Config &config);

uint32_t parallel_compute_helper(const uint8_t *data,
                                 size_t length, uint8_t width,
                                 uint32_t **T, uint32_t initial_crc);

uint32_t reflect(uint32_t data, uint8_t width);

#endif // CRC_H
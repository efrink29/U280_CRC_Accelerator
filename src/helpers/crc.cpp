#include "crc.h"
#include <cstring>

uint32_t reflect(uint32_t data, uint8_t width)
{
    uint32_t reflection = 0;
    for (uint8_t bit = 0; bit < width; ++bit)
    {
        if (data & 0x01)
        {
            reflection |= (1 << ((width - 1) - bit));
        }
        data >>= 1;
    }
    return reflection;
}

uint32_t *create_standard_table(const CRC_Config &config)
{
    const uint32_t w = config.width;
    const uint64_t mask64 = (w == 32) ? 0xFFFFFFFFull : ((1ull << w) - 1ull);

    const uint32_t mask = static_cast<uint32_t>(mask64);

    if (w < 8)
        return nullptr;

    uint32_t *table = new uint32_t[256];

    const uint32_t poly = (config.polynomial & mask);
    if (config.reflect_input)
    {

        const uint32_t rpoly = reflect(poly, w) & mask;

        for (uint32_t i = 0; i < 256; ++i)
        {
            uint32_t crc = i;
            for (int j = 0; j < 8; ++j)
            {
                if (crc & 1u)
                    crc = (crc >> 1) ^ rpoly;
                else
                    crc >>= 1;
            }
            table[i] = crc & mask;
        }
    }
    else
    {

        for (uint32_t i = 0; i < 256; ++i)
        {
            uint32_t crc = (i << (w - 8)) & mask;
            for (int j = 0; j < 8; ++j)
            {
                if (crc & (1u << (w - 1)))
                    crc = ((crc << 1) ^ poly) & mask;
                else
                    crc = (crc << 1) & mask;
            }
            table[i] = crc & mask;
        }
    }

    return table;
}

uint32_t standard_compute(const uint8_t *data,
                          size_t length,
                          const CRC_Config &config, bool table_gen)
{
    const uint32_t w = config.width;
    const uint64_t mask64 = (w == 32) ? 0xFFFFFFFFull : ((1ull << w) - 1ull);
    const uint32_t mask = static_cast<uint32_t>(mask64);

    if (w < 8)
    {

        return 0;
    }

    uint32_t *table = create_standard_table(config);
    if (!table)
        return 0;

    uint32_t crc = config.initial_value & mask;

    if (config.reflect_input)
    {

        for (size_t i = 0; i < length; ++i)
        {
            uint8_t index = static_cast<uint8_t>((crc ^ data[i]) & 0xFF);
            crc = (crc >> 8) ^ table[index];
        }
    }
    else
    {

        for (size_t i = 0; i < length; ++i)
        {
            uint8_t index = static_cast<uint8_t>(((crc >> (w - 8)) ^ data[i]) & 0xFF);
            crc = ((crc << 8) & mask) ^ table[index];
        }
    }

    delete[] table;

    if (table_gen)
    {
        return crc & mask;
    }
    if (!config.reflect_output)
    {
        crc = reflect(crc, w) & mask;
    }

    crc ^= (config.final_xor_value & mask);
    return crc & mask;
}

uint32_t standard_compute_golden(const uint8_t *data,
                                 size_t length,
                                 const CRC_Config &config)
{
    const uint32_t w = config.width;
    const uint64_t mask64 = (w == 32) ? 0xFFFFFFFFull : ((1ull << w) - 1ull);
    const uint32_t mask = static_cast<uint32_t>(mask64);

    if (w < 8)
    {

        return 0;
    }

    uint32_t *table = create_standard_table(config);
    if (!table)
        return 0;

    uint32_t crc = config.initial_value & mask;

    if (config.reflect_input)
    {

        for (size_t i = 0; i < length; ++i)
        {
            uint8_t index = static_cast<uint8_t>((crc ^ data[i]) & 0xFF);
            crc = (crc >> 8) ^ table[index];
        }
    }
    else
    {

        for (size_t i = 0; i < length; ++i)
        {
            uint8_t index = static_cast<uint8_t>(((crc >> (w - 8)) ^ data[i]) & 0xFF);
            crc = ((crc << 8) & mask) ^ table[index];
        }
    }

    delete[] table;

    if (!config.reflect_output)
    {
        crc = reflect(crc, w) & mask;
    }

    crc ^= (config.final_xor_value & mask);
    return crc & mask;
}

uint32_t **create_parallel_tables(const CRC_Config &config)
{
    const uint32_t w = config.width;
    if (w < 8)
        return nullptr;

    const uint64_t mask64 = (w == 32) ? 0xFFFFFFFFull : ((1ull << w) - 1ull);
    const uint32_t mask = static_cast<uint32_t>(mask64);

    // T[0] = standard table in the correct processing direction
    uint32_t **T = new uint32_t *[16];
    CRC_Config tmp_config = config;
    if (!config.reflect_input)
    {
        tmp_config.reflect_input = true;
    }

    // Allocate the rest
    for (int i = 0; i < 16; ++i)
    {
        T[i] = new uint32_t[256];
    }

    for (int i = 0; i < 16; i++)
    {
        for (int x = 0; x < 256; ++x)
        {
            uint8_t *tmp = new uint8_t[16];
            tmp[0] = x & 0xFF;
            for (int k = 0; k < 16; ++k)
            {
                if (i == k)
                {

                    tmp[k] = x & 0xFF;
                }
                else
                {
                    tmp[k] = 0;
                }
            }
            uint32_t v = standard_compute(reinterpret_cast<uint8_t *>(tmp), 16, tmp_config, true) & mask;

            T[15 - i][x] = v;
            delete[] tmp;
        }
    }
    if (!config.reflect_input)
    {
        uint32_t **reflected_T = new uint32_t *[16];
        for (int i = 0; i < 16; ++i)
        {
            reflected_T[i] = new uint32_t[256];
            if (i > 11)
            {
                for (int j = 0; j < 256; ++j)
                {
                    reflected_T[i][j] = T[i][j];
                }
            }
            else
            {
                for (int j = 0; j < 256; ++j)
                {
                    uint8_t reflected_j = static_cast<uint8_t>(reflect(j, 8) & 0xFF);

                    reflected_T[i][j] = T[i][reflected_j];
                }
            }
        }
        for (int i = 0; i < 16; ++i)
        {
            delete[] T[i];
        }
        delete[] T;
        T = reflected_T;
    }

    return T;
}

void print_byte(uint8_t byte)
{
    uint8_t high_nibble = (byte >> 4) & 0x0F;
    uint8_t low_nibble = byte & 0x0F;

    if (high_nibble < 10)
        std::cout << static_cast<char>('0' + high_nibble);
    else
        std::cout << static_cast<char>('A' + (high_nibble - 10));

    if (low_nibble < 10)
        std::cout << static_cast<char>('0' + low_nibble);
    else
        std::cout << static_cast<char>('A' + (low_nibble - 10));
}

uint32_t parallel_compute(const uint8_t *data,
                          size_t length,
                          const CRC_Config &config)
{
    const uint32_t w = config.width;
    if (w < 8)
    {
        // For w < 8, use a bitwise implementation instead of slicing-by-16.
        return standard_compute(data, length, config);
    }

    uint8_t *tmp_data = new uint8_t[length];
    memcpy(tmp_data, data, length);

    const uint64_t mask64 = (w == 32) ? 0xFFFFFFFFull : ((1ull << w) - 1ull);
    const uint32_t mask = static_cast<uint32_t>(mask64);
    CRC_Config tmp_config = config;

    uint32_t **T = create_parallel_tables(tmp_config);
    if (!T)
        return 0;

    if (!config.reflect_input)
    {
        // If not reflected, we need to reflect the data first
        for (size_t i = 0; i < length; i++)
        {
            if (i % 16 <= 3)
            {
                tmp_data[i] = static_cast<uint8_t>(reflect(tmp_data[i], 8) & 0xFF);
            }
        }
    }
    uint32_t crc = (config.initial_value & mask);

    // Reflected (LSB-first) slicing-by-16
    while (length >= 16)
    {
        // Fold the first 4 bytes into CRC (little-endian)

        uint8_t byte0 = tmp_data[0];
        uint8_t byte1 = tmp_data[1];
        uint8_t byte2 = tmp_data[2];
        uint8_t byte3 = tmp_data[3];

        uint8_t crc0 = static_cast<uint8_t>(crc & 0xFF);
        uint8_t crc1 = static_cast<uint8_t>((crc >> 8) & 0xFF);
        uint8_t crc2 = static_cast<uint8_t>((crc >> 16) & 0xFF);
        uint8_t crc3 = static_cast<uint8_t>((crc >> 24) & 0xFF);

        print_byte(byte0);
        print_byte(byte1);
        print_byte(byte2);
        print_byte(byte3);
        std::cout << " ^ ";
        print_byte(crc0);
        print_byte(crc1);
        print_byte(crc2);
        print_byte(crc3);
        std::cout << " => ";

        byte0 = byte0 ^ crc0;
        byte1 = byte1 ^ crc1;
        byte2 = byte2 ^ crc2;
        byte3 = byte3 ^ crc3;

        print_byte(byte0);
        print_byte(byte1);
        print_byte(byte2);
        print_byte(byte3);
        std::cout << std::endl;

        crc =
            T[15][tmp_data[15]] ^ T[14][tmp_data[14]] ^ T[13][tmp_data[13]] ^ T[12][tmp_data[12]] ^
            T[11][tmp_data[11]] ^ T[10][tmp_data[10]] ^ T[9][tmp_data[9]] ^ T[8][tmp_data[8]] ^
            T[7][tmp_data[7]] ^ T[6][tmp_data[6]] ^ T[5][tmp_data[5]] ^ T[4][tmp_data[4]] ^
            T[3][byte3] ^
            T[2][byte2] ^
            T[1][byte1] ^
            T[0][byte0];

        tmp_data += 16;
        length -= 16;

        crc &= mask;
    }
    if (!config.reflect_input)
    {
        tmp_config.reflect_input = true;
    }
    uint32_t *std_table = create_standard_table(tmp_config);

    // Tail (byte-at-a-time)

    for (size_t i = 0; i < length; ++i)
    {
        uint8_t index = static_cast<uint8_t>((crc ^ tmp_data[i]) & 0xFF);
        crc = (crc >> 8) ^ std_table[index];
    }

    // Free tables
    for (int i = 0; i < 16; ++i)
    {
        delete[] T[i];
    }
    delete[] T;

    // Post-processing

    crc ^= (config.final_xor_value & mask);
    if (!config.reflect_output)
    {
        crc = reflect(crc, w) & mask;
    }
    return crc & mask;
}

uint32_t parallel_compute_helper(const uint8_t *data,
                                 size_t length, uint8_t width,
                                 uint32_t **T,
                                 uint32_t initial_crc)
{
    uint32_t crc = initial_crc;
    while (length >= 16)
    {
        // Fold the first 4 bytes into CRC (little-endian)
        uint32_t c =
            crc ^
            (static_cast<uint32_t>(data[0]) |
             (static_cast<uint32_t>(data[1]) << 8) |
             (static_cast<uint32_t>(data[2]) << 16) |
             (static_cast<uint32_t>(data[3]) << 24));

        crc =
            T[15][data[15]] ^ T[14][data[14]] ^ T[13][data[13]] ^ T[12][data[12]] ^
            T[11][data[11]] ^ T[10][data[10]] ^ T[9][data[9]] ^ T[8][data[8]] ^
            T[7][data[7]] ^ T[6][data[6]] ^ T[5][data[5]] ^ T[4][data[4]] ^
            T[3][(c >> 24) & 0xFF] ^
            T[2][(c >> 16) & 0xFF] ^
            T[1][(c >> 8) & 0xFF] ^
            T[0][(c) & 0xFF];

        data += 16;
        length -= 16;
    }
    return crc;
}
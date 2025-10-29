#include <CL/cl.h>
#include <cstdint>
#include <cstdio>
#include <ios>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "cmdlineparser.h"
#include "xcl2.hpp"
#include "manager.hpp"

#include <unistd.h>
#include <string>
#include <chrono>

#define DATA_SIZE 32
#define BLOCK_SIZE 16
#define TABLE_SIZE 256

// #define SETUP_KERNEL(knum)                                                                                                                          \
//     HBM_ext.banks = (knum * 2) | XCL_MEM_TOPOLOGY;                                                                                                  \
//     OCL_CHECK(err, cl::Buffer input_buffer_A##knum(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, buf_size_bytes, &HBM_ext, &err));             \
//     OCL_CHECK(err, cl::Buffer input_buffer_B##knum(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, buf_size_bytes, &HBM_ext, &err));             \
//     OCL_CHECK(err, cl::Buffer output_buffer_A##knum(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, buf_size_bytes, &HBM_ext, &err));           \
//     OCL_CHECK(err, cl::Buffer output_buffer_B##knum(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, buf_size_bytes, &HBM_ext, &err));           \
//                                                                                                                                                     \
//     HBM_ext.banks = ((knum * 2) + 1) | XCL_MEM_TOPOLOGY;                                                                                            \
//     OCL_CHECK(err, cl::Buffer table_buffer_##knum(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 256 * 16 * sizeof(uint32_t), &HBM_ext, &err)); \
//                                                                                                                                                     \
//     kComps[knum].inBufferA = input_buffer_A##knum;                                                                                                  \
//     kComps[knum].inBufferB = input_buffer_B##knum;                                                                                                  \
//     kComps[knum].outBufferA = output_buffer_A##knum;                                                                                                \
//     kComps[knum].outBufferB = output_buffer_B##knum;                                                                                                \
//     kComps[knum].tableBuffer = table_buffer_##knum;

// #define SETUP_SIXTEEN_KERNELS \
//     SETUP_KERNEL(0);          \
//     SETUP_KERNEL(1);          \
//     SETUP_KERNEL(2);          \
//     SETUP_KERNEL(3);          \
//     SETUP_KERNEL(4);          \
//     SETUP_KERNEL(5);          \
//     SETUP_KERNEL(6);          \
//     SETUP_KERNEL(7);          \
//     SETUP_KERNEL(8);          \
//     SETUP_KERNEL(9);
// // SETUP_KERNEL(10);     \
// // SETUP_KERNEL(11);     \
// // SETUP_KERNEL(12);     \
// // SETUP_KERNEL(13);     \
// // SETUP_KERNEL(14);     \
// // SETUP_KERNEL(15);

// #define PROGRAM_KERNEL(knum)                                                   \
//     kernelConfigString = "calculate_crc:{CRC_" + std::to_string(knum) + "}\0"; \
//     charBuf = std::strcpy(charBuf, kernelConfigString.c_str());                \
//     OCL_CHECK(err, kComps[knum].kernel = cl::Kernel(program, charBuf, &err));

// #define PROGRAM_TEN_KERNELS \
//     PROGRAM_KERNEL(0);      \
//     PROGRAM_KERNEL(1);      \
//     PROGRAM_KERNEL(2);      \
//     PROGRAM_KERNEL(3);      \
//     PROGRAM_KERNEL(4);      \
//     PROGRAM_KERNEL(5);      \
//     PROGRAM_KERNEL(6);      \
//     PROGRAM_KERNEL(7);      \
//     PROGRAM_KERNEL(8);      \
//     PROGRAM_KERNEL(9);

typedef std::vector<unsigned char> Bytes;

uint32_t maskCRC(uint32_t value, int width)
{
    if (width == 32)
    {
        return value;
    }
    return value & ((1u << width) - 1);
}

uint32_t reverseBits(uint32_t value, int width)
{
    uint32_t reversed = 0;
    for (int i = 0; i < width; ++i)
    {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    return maskCRC(reversed, width);
}

static uint32_t reflect(uint32_t data, uint8_t width)
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

uint32_t *generateStandardCRCTable(KernelConfig config)
{
    const uint32_t w = config.crcWidth;
    const uint64_t mask64 = (w == 32) ? 0xFFFFFFFFull : ((1ull << w) - 1ull);
    const uint32_t mask = static_cast<uint32_t>(mask64);

    if (w < 8)
        return nullptr;

    uint32_t *table = new uint32_t[256];

    const uint32_t poly = (config.polynomial & mask);
    if (config.refInput)
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

uint32_t standardCompute(const uint8_t *data,
                         size_t length,
                         const KernelConfig &config)
{
    const uint32_t w = config.crcWidth;
    const uint64_t mask64 = (w == 32) ? 0xFFFFFFFFull : ((1ull << w) - 1ull);
    const uint32_t mask = static_cast<uint32_t>(mask64);

    if (w < 8)
    {

        return 0;
    }

    uint32_t *table = generateStandardCRCTable(config);
    if (!table)
        return 0;

    uint32_t crc = config.init_val & mask;

    if (config.refInput)
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

    if (config.refOutput)
    {
        crc = reflect(crc, w) & mask;
    }

    crc ^= (config.xor_out & mask);
    return crc & mask;
}

uint32_t **generateParallelCRCTables(KernelConfig config)
{
    const uint32_t w = config.crcWidth;
    if (w < 8)
        return nullptr;

    const uint64_t mask64 = (w == 32) ? 0xFFFFFFFFull : ((1ull << w) - 1ull);
    const uint32_t mask = static_cast<uint32_t>(mask64);

    // T[0] = standard table in the correct processing direction
    uint32_t **T = new uint32_t *[16];
    T[0] = generateStandardCRCTable(config); // already MSB- or LSB-first per reflect_input
    if (!T[0])
    {
        delete[] T;
        return nullptr;
    }

    // Allocate the rest
    for (int i = 1; i < 16; ++i)
    {
        T[i] = new uint32_t[256];
    }

    if (config.refInput)
    {
        // Reflected (LSB-first):
        //   T[i][x] = (T[i-1][x] >> 8) ^ T[0][ T[i-1][x] & 0xFF ]
        for (int i = 0; i < 16; i++)
        {
            for (int x = 0; x < 256; ++x)
            {
                uint8_t *tmp = new uint8_t[16];
                for (int k = 0; k < 16; ++k)
                {
                    if (i == k)
                    {
                        tmp[k] = x;
                    }
                    else
                    {
                        tmp[k] = 0;
                    }
                }
                uint32_t v = standardCompute(reinterpret_cast<uint8_t *>(tmp), 16, config);
                T[i][x] = v;
                delete[] tmp;
            }
        }
    }
    else
    {
        // Non-reflected (MSB-first):
        //   T[i][x] = ((T[i-1][x] << 8) & mask) ^ T[0][ (T[i-1][x] >> (w-8)) & 0xFF ]
        for (int i = 0; i < 16; i++)
        {
            for (int x = 0; x < 256; ++x)
            {
                uint8_t *tmp = new uint8_t[16];
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
                uint32_t v = standardCompute(reinterpret_cast<uint8_t *>(tmp), 16, config);
                T[i][x] = v;
                delete[] tmp;
            }
        }
    }

    return T;
}

void reflectInput(unsigned char *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        (data)[i] = (reverseBits((data)[i], 8) & 0xFF);
    }
}

void load_data_chunk(unsigned char *data, std::ifstream &inputfile, size_t size)
{
    inputfile.read(reinterpret_cast<char *>(data), size);
}

void save_to_file(uint32_t *data, std::ofstream &outputFile, size_t size, bool reflect, int crcSize)
{
    if (reflect)
    {
        for (int i = 0; i < size; i++)
        {
            data[i] = reverseBits(data[i], crcSize);
        }
    }
    outputFile.write(reinterpret_cast<const char *>(data), size * sizeof(uint32_t));
}

void loadConfig(std::string filename, KernelConfig &cfg)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }
    file >> cfg.crcWidth;
    file >> std::hex >> cfg.polynomial;
    file >> std::hex >> cfg.init_val;
    file >> cfg.refInput;
    cfg.refInput = !cfg.refInput;
    file >> cfg.refOutput;
    cfg.refOutput = !cfg.refOutput;
    file >> std::hex >> cfg.xor_out;
    file >> std::dec >> cfg.chunkSize;
}

void printData(const Bytes &data)
{
    if (data.size() == 0)
    {
        std::cout << "<empty data>" << std::endl;
        return;
    }
    if (data.size() > 64)
    {
        for (size_t i = 0; i < 32; i++)
        {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]) << " ";
        }
        std::cout << "... ";
        for (size_t i = data.size() - 32; i < data.size(); i++)
        {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]) << " ";
        }
        std::cout << std::endl;
        return;
    }
    for (uint8_t byte : data)
    {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte) << " ";
    }
    std::cout << std::endl;
}

std::vector<uint32_t> fast_crc(const Bytes &data, const KernelConfig &cfg)
{
    uint32_t *stdTbl = generateStandardCRCTable(cfg);

    uint32_t crc = maskCRC(cfg.init_val, cfg.crcWidth);
    size_t chunkBytes = static_cast<size_t>(cfg.chunkSize);
    if (chunkBytes == 0)
        throw std::runtime_error("chunkSize must be > 0");
    size_t totalBytes = data.size();
    size_t nChunks = (totalBytes + chunkBytes - 1) / chunkBytes;

    std::vector<uint32_t> results(nChunks, 0);

    for (size_t chunk = 0; chunk < nChunks; ++chunk)
    {
        size_t offset = chunk * chunkBytes;
        size_t bytesToProcess = std::min(chunkBytes, totalBytes - offset);

        std::vector<unsigned char, aligned_allocator<unsigned char>> chunkData(bytesToProcess, 0);
        std::copy(data.begin() + offset, data.begin() + offset + bytesToProcess, chunkData.begin());

        crc = standardCompute(chunkData.data(), bytesToProcess, cfg);

        results[chunk] = crc;
    }

    return results;
}

struct KernelComponents
{

    // Host Memory Pointers
    unsigned char *hostInA;
    unsigned char *hostInB;
    uint32_t *hostOutA;
    uint32_t *hostOutB;

    // Command Queues
    cl::CommandQueue queueA;
    cl::CommandQueue queueB;
    cl::CommandQueue queueK;

    // Kernel Object
    cl::Kernel kernel;

    // Input Buffers (FPGA Memory Channels)
    cl::Buffer inBufferA;
    cl::Buffer inBufferB;

    // Output Buffers
    cl::Buffer outBufferA;
    cl::Buffer outBufferB;

    // Table Buffer
    cl::Buffer tableBuffer;
};

void printConfiguration(std::vector<KernelConfig *> configs)
{
    int numKernels = 10 / configs.size();
    if (configs.size() == 0)
    {
        std::cout << "No Configurations Loaded!" << std::endl;
        return;
    }

    for (int i = 0; i < configs.size(); i++)
    {
        std::cout << "Config " << i << std::endl;
        std::cout << "Polynomial: " << std::hex << configs[i]->polynomial << std::endl;
        std::cout << "Init Value: " << std::hex << configs[i]->init_val << std::endl;
        std::cout << "XOR Out: " << std::hex << configs[i]->xor_out << std::endl;
        std::cout << "Reflect Input: " << std::boolalpha << configs[i]->refInput << std::endl;
        std::cout << "Reflect Output: " << std::boolalpha << configs[i]->refOutput << std::endl;
        std::cout << "CRC Width: " << std::dec << configs[i]->crcWidth << std::endl;
        std::cout << "Chunk Size: " << std::dec << configs[i]->chunkSize << std::endl;
        std::cout << std::endl;
    }
    int k = 0;
    int c = 0;
    for (int i = 0; i < 16; i++)
    {
        if (c >= configs.size())
        {
            std::cout << "Kernel" << i << " Not in use" << std::endl;
        }
        else
        {
            std::cout << "Kernel" << i << " Config: " << c << std::endl;
        }

        k++;
        if (k == numKernels)
        {
            k = 0;
            c++;
        }
    }
}

std::vector<int64_t> calc_fpga_sequential(FpgaManager &mgr,
                                          const Bytes &data1, const KernelConfig &crc32,
                                          const Bytes &data2, const KernelConfig &crc16, int numProcesses)
{
    std::cout << "Calculating Sequentially on FPGA..." << std::endl;
    std::vector<int64_t> durations;
    bool use32 = true;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto mid_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numProcesses; i++)
    {
        if (use32)
        {
            auto res1 = mgr.calculate_crc(data1, crc32);
        }
        else
        {
            auto res2 = mgr.calculate_crc(data2, crc16);
        }
        use32 = !use32;
        start_time = mid_time;
        mid_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(mid_time - start_time).count();
        durations.push_back(duration);
    }
    return durations;
}

std::vector<int64_t> calc_fpga_parallel(FpgaManager &mgr,
                                        const Bytes &data1, const KernelConfig &crc32,
                                        const Bytes &data2, const KernelConfig &crc16, int numProcesses)
{
    std::cout << "Calculating in Parallel on FPGA..." << std::endl;
    std::vector<int64_t> durations;
    bool use32 = true;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto mid_time = std::chrono::high_resolution_clock::now();
    std::vector<std::future<std::vector<uint32_t>>> futures;
    // Start all tasks
    for (int i = 0; i < numProcesses; i++)
    {
        if (use32)
        {
            futures.push_back(mgr.submit(data1, crc32));
        }
        else
        {
            futures.push_back(mgr.submit(data2, crc16));
        }
        use32 = !use32;
    }
    // Finish all tasks
    for (int i = 0; i < numProcesses; i++)
    {
        auto res = futures[i].get();
        start_time = mid_time;
        mid_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(mid_time - start_time).count();
        durations.push_back(duration);
    }
    return durations;
}

std::vector<int64_t> calc_host_sequential(const Bytes &data1, const KernelConfig &crc32,
                                          const Bytes &data2, const KernelConfig &crc16, int numProcesses)
{
    std::cout << "Calculating Sequentially on Host..." << std::endl;
    std::vector<int64_t> durations;
    bool use32 = true;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto mid_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numProcesses; i++)
    {
        if (use32)
        {
            auto res1 = fast_crc(data1, crc32);
        }
        else
        {
            auto res2 = fast_crc(data2, crc16);
        }
        use32 = !use32;
        start_time = mid_time;
        mid_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(mid_time - start_time).count();
        durations.push_back(duration);
    }
    return durations;
}

std::vector<int64_t> calc_host_parallel(const Bytes &data1, const KernelConfig &crc32,
                                        const Bytes &data2, const KernelConfig &crc16, int numProcesses)
{
    std::cout << "Calculating in Parallel on Host..." << std::endl;
    std::vector<int64_t> durations;
    bool use32 = true;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto mid_time = std::chrono::high_resolution_clock::now();
    std::vector<std::future<std::vector<uint32_t>>> futures;
    // Start all tasks
    for (int i = 0; i < numProcesses; i++)
    {
        if (use32)
        {
            futures.push_back(std::async(std::launch::async, fast_crc, data1, crc32));
        }
        else
        {
            futures.push_back(std::async(std::launch::async, fast_crc, data2, crc16));
        }
        use32 = !use32;
    }
    // Finish all tasks
    for (int i = 0; i < numProcesses; i++)
    {
        auto res = futures[i].get();
        start_time = mid_time;
        mid_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(mid_time - start_time).count();
        durations.push_back(duration);
    }
    return durations;
}

void printFourByteHex(const std::vector<uint32_t> &data)
{
    for (auto d : data)
    {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << d << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{

    // Parse command line arguments
    sda::utils::CmdLineParser parser;

    parser.addSwitch("--xclbin_file", "-x", "Binary file string", "");
    parser.addSwitch("--frequency", "-f", "Operating frequency, in MHz", "300");
    parser.addSwitch("--buf_size_mb", "-m", "Test buffer size, in MB", "16");
    parser.addSwitch("--buf_size_kb", "-k", "Test buffer size, in KB", "0");
    parser.addSwitch("--num_compute_units", "-n", "Number of compute units to use (max 10)", "2");
    parser.addSwitch("--filename", "-o", "Output filename", "output.dat");
    parser.addSwitch("--test_mode", "-t", "Test mode (0=static, 1=dynamic)", "0");
    parser.parse(argc, argv);

    std::string xclbinFile = parser.value("xclbin_file");

    float frequency = stof(parser.value("frequency"));
    int64_t buf_size_mb = stoi(parser.value("buf_size_mb"));
    int64_t buf_size_kb = stoi(parser.value("buf_size_kb"));
    int num_workers = stoi(parser.value("num_compute_units"));
    std::string outputFileName = parser.value("filename");
    int test_mode = stoi(parser.value("test_mode"));

    int max_compute_units = 14;

    num_workers = std::min(num_workers, max_compute_units);

    if (xclbinFile.empty())
    {
        parser.printHelp();
        return EXIT_FAILURE;
    }
    if (access(xclbinFile.c_str(), R_OK) != 0)
    {
        std::cerr << "ERROR: " << xclbinFile << " file not found" << std::endl;
        parser.printHelp();
        return EXIT_FAILURE;
    }

    if (buf_size_kb == 0)
    {
        buf_size_kb = buf_size_mb * 1024;
    }
    int64_t buf_size_bytes = buf_size_kb * 1024;
    buf_size_bytes -= buf_size_bytes % 16;

    KernelConfig crc32;
    loadConfig("Configs/CRC_32", crc32);
    KernelConfig crc16;
    loadConfig("Configs/CRC_16", crc16);

    std::vector<unsigned char> data1(65536);
    std::vector<unsigned char> data2(65536);
    for (size_t i = 0; i < data1.size(); i++)
    {
        data1[i] = static_cast<unsigned char>(0x42 + (i % 256));
    }
    FpgaManager mgr("kernel.xclbin", buf_size_bytes, max_compute_units, num_workers);
    auto res1a = mgr.calculate_crc(data1, crc32);

    // Delay 100ms
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // open output file

    // Test Independent Run Time

    auto stdTbl = generateStandardCRCTable(crc32);
    for (int i = 0; i < 256; i++)
    {

        printFourByteHex(std::vector<uint32_t>{stdTbl[i]});
        if (i % 8 == 7)
            std::cout << std::endl;
    }
    auto resFPGA = mgr.calculate_crc(data1, crc32);

    uint32_t uin = standardCompute(data1.data(), data1.size(), crc32);
    std::vector<uint32_t> resHost;
    resHost.push_back(uin);
    std::cout << "Input Data:" << std::endl;
    printData(data1);
    std::cout << "Results:" << std::endl;
    for (size_t i = 0; i < resHost.size(); i++)
    {
        std::cout << "Host: " << std::hex << resHost[i] << " | " << reverseBits(resHost[i], 32) << " FPGA: " << std::hex << resFPGA[i] << " | " << reverseBits(resFPGA[i], 32) << std::endl;
    }

    return 0;
}

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
#include "helpers/crc.h"

#include <unistd.h>
#include <string>
#include <chrono>

#define DATA_SIZE 32
#define BLOCK_SIZE 16
#define TABLE_SIZE 256

typedef std::vector<unsigned char> Bytes;

void load_data_chunk(unsigned char *data, std::ifstream &inputfile, size_t size)
{
    inputfile.read(reinterpret_cast<char *>(data), size);
}

void save_to_file(uint32_t *data, std::ofstream &outputFile, size_t size, bool ref, int crcSize)
{
    if (ref)
    {
        for (size_t i = 0; i < size; i++)
        {
            data[i] = reflect(data[i], crcSize);
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
    // cfg.refInput = !cfg.refInput;
    file >> cfg.refOutput;
    // cfg.refOutput = !cfg.refOutput;
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
    CRC_Config crc_cfg;
    crc_cfg.polynomial = cfg.polynomial;
    crc_cfg.initial_value = cfg.init_val;
    crc_cfg.final_xor_value = cfg.xor_out;
    crc_cfg.reflect_input = cfg.refInput;
    crc_cfg.reflect_output = cfg.refOutput;
    crc_cfg.width = static_cast<uint8_t>(cfg.crcWidth);
    crc_cfg.chunk_size = static_cast<size_t>(cfg.chunkSize);
    uint32_t *stdTbl = create_standard_table(crc_cfg);
    uint32_t mask = (cfg.crcWidth == 32) ? 0xFFFFFFFF : ((1u << cfg.crcWidth) - 1u);
    uint32_t crc = cfg.init_val & mask;
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
        // crc_cfg.initial_value = crc;
        crc = standard_compute(chunkData.data(), bytesToProcess, crc_cfg) & mask;

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
                                        const Bytes &data2, const KernelConfig &crc16, int numProcesses, int numWorkers)
{
    std::cout << "Calculating in Parallel on Host..." << std::endl;
    std::vector<int64_t> durations;
    bool use32 = true;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto mid_time = std::chrono::high_resolution_clock::now();
    std::vector<std::future<std::vector<uint32_t>>> futures;
    // Start all tasks

    int completed_tasks = 0;
    while (completed_tasks < numProcesses)
    {
        int tasks_to_launch = std::min(numWorkers, numProcesses - completed_tasks);
        for (int i = 0; i < tasks_to_launch; i++)
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
            completed_tasks++;
        }
        // Finish launched tasks
        for (int i = futures.size() - tasks_to_launch; i < futures.size(); i++)
        {
            auto res = futures[i].get();
            start_time = mid_time;
            mid_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(mid_time - start_time).count();
            durations.push_back(duration);
        }
    }
    return durations;
}

void printFourByteHex(const std::vector<uint32_t> &data)
{
    for (auto d : data)
    {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << d << " ";
    }
    std::cout << std::dec << std::endl;
}

double get_throughput(size_t data_size_bytes, int64_t duration_microseconds)
{
    if (duration_microseconds == 0)
        return 0;
    double duration_seconds = static_cast<double>(duration_microseconds) / 1e6;
    double throughput = static_cast<double>(data_size_bytes) / duration_seconds; // bytes per second
    return throughput / (1024 * 1024);                                           // MB/s
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

    int max_compute_units = 16;

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
    std::vector<KernelConfig *> configs;
    configs.push_back(&crc32);
    // printConfiguration(configs);
    // crc32.chunkSize = 32;
    KernelConfig crc16;
    loadConfig("Configs/CRC_16", crc32);

    std::vector<unsigned char> data1(6553600);
    std::vector<unsigned char> data2(655360);

    for (size_t i = 0; i < data1.size(); i++)
    {
        data1[i] = static_cast<unsigned char>((i + (i / 65536)) & 0xFF);
    }
    // std::memcpy(data1.data(), testStr1, 32);
    std::cout << data1.size() << std::endl;
    crc32.refInput = true;
    std::vector<KernelConfig> cfgs = {crc32};

    FpgaManager mgr(xclbinFile, buf_size_bytes, max_compute_units, num_workers, cfgs);
    auto res1a = mgr.calculate_crc(data1, crc32);

    // Delay 100ms
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // open output file

    // Test Independent Run Time

    CRC_Config crc_cfg;
    crc_cfg.polynomial = crc32.polynomial;
    crc_cfg.initial_value = crc32.init_val;
    crc_cfg.final_xor_value = crc32.xor_out;
    crc_cfg.reflect_input = crc32.refInput;
    crc_cfg.reflect_output = crc32.refOutput;
    crc_cfg.width = static_cast<uint8_t>(crc32.crcWidth);
    crc_cfg.chunk_size = static_cast<size_t>(crc32.chunkSize);
    auto stdTbl = create_standard_table(crc_cfg);
    // for (int i = 0; i < 256; i++)
    // {

    //     printFourByteHex(std::vector<uint32_t>{stdTbl[i]});
    //     if (i % 8 == 7)
    //         std::cout << std::endl;
    // }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto resFPGA = mgr.calculate_crc(data1, crc32);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    double throughput = get_throughput(data1.size(), duration);
    std::cout << "FPGA CRC32 Time: " << duration << " us, Throughput: " << throughput << " MB/s" << std::endl;
    crc_cfg.polynomial = crc32.polynomial;
    crc_cfg.initial_value = crc32.init_val;
    crc_cfg.final_xor_value = crc32.xor_out;
    crc_cfg.reflect_input = crc32.refInput;
    crc_cfg.reflect_output = crc32.refOutput;
    crc_cfg.width = static_cast<uint8_t>(crc32.crcWidth);
    crc_cfg.chunk_size = static_cast<size_t>(crc32.chunkSize);

    start_time = std::chrono::high_resolution_clock::now();
    auto resHost = fast_crc(data1, crc32);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    throughput = get_throughput(data1.size(), duration);
    std::cout << "Host CRC32 Time: " << duration << " us, Throughput: " << throughput << " MB/s" << std::endl;
    // std::vector<uint32_t> resHost;
    // resHost.push_back(uin);
    std::cout << "Input Data:" << std::endl;
    printData(data1);
    int numTests = 10;

    // auto sequentialFPGA = calc_fpga_sequential(mgr, data1, crc32, data2, crc32, numTests);
    // auto parallelFPGA = calc_fpga_parallel(mgr, data1, crc32, data2, crc32, numTests);
    // auto sequentialHost = calc_host_sequential(data1, crc32, data2, crc32, numTests);
    // auto parallelHost = calc_host_parallel(data1, crc32, data2, crc32, numTests, num_workers);
    // std::cout << std::dec;
    // auto totalDataSize = (data1.size() + data2.size()) * (numTests / 2);
    // int64_t totalTimeSeqFPGA = 0;
    // for (auto t : sequentialFPGA)
    //     totalTimeSeqFPGA += t;
    // int64_t totalTimeParFPGA = 0;
    // for (auto t : parallelFPGA)
    //     totalTimeParFPGA += t;
    // int64_t totalTimeSeqHost = 0;
    // for (auto t : sequentialHost)
    //     totalTimeSeqHost += t;
    // int64_t totalTimeParHost = 0;
    // for (auto t : parallelHost)
    //     totalTimeParHost += t;
    // double seqFpgThru = get_throughput(totalDataSize, totalTimeSeqFPGA);
    // double parFpgThru = get_throughput(totalDataSize, totalTimeParFPGA);
    // double seqHostThru = get_throughput(totalDataSize, totalTimeSeqHost);
    // double parHostThru = get_throughput(totalDataSize, totalTimeParHost);
    // std::cout << "Sequential FPGA Throughput: " << seqFpgThru << " MB/s" << std::endl;
    // std::cout << "Parallel FPGA Throughput: " << parFpgThru << " MB/s" << std::endl;
    // std::cout << "Sequential Host Throughput: " << seqHostThru << " MB/s" << std::endl;
    // std::cout << "Parallel Host Throughput: " << parHostThru << " MB/s" << std::endl;
    // std::cout << "Results:" << std::endl;
    for (size_t i = 0; i < resHost.size() && i < 5; i++)
    {
        std::cout << "Host: " << std::hex << resHost[i] << " | " << reflect(resHost[i], 16) << " FPGA: " << std::hex << resFPGA[i] << " | " << reflect(resFPGA[i], 16) << std::endl;
    }

    return 0;
}

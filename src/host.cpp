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
#include <unistd.h>
#include <string>
#include <chrono>

#define DATA_SIZE 32
#define BLOCK_SIZE 16
#define TABLE_SIZE 256

#define SETUP_KERNEL(knum)                                                                                                                          \
    HBM_ext.banks = (knum * 2) | XCL_MEM_TOPOLOGY;                                                                                                  \
    OCL_CHECK(err, cl::Buffer input_buffer_A##knum(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, buf_size_bytes, &HBM_ext, &err));             \
    OCL_CHECK(err, cl::Buffer input_buffer_B##knum(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, buf_size_bytes, &HBM_ext, &err));             \
    OCL_CHECK(err, cl::Buffer output_buffer_A##knum(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, buf_size_bytes, &HBM_ext, &err));           \
    OCL_CHECK(err, cl::Buffer output_buffer_B##knum(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, buf_size_bytes, &HBM_ext, &err));           \
                                                                                                                                                    \
    HBM_ext.banks = ((knum * 2) + 1) | XCL_MEM_TOPOLOGY;                                                                                            \
    OCL_CHECK(err, cl::Buffer table_buffer_##knum(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 256 * 16 * sizeof(uint32_t), &HBM_ext, &err)); \
                                                                                                                                                    \
    kComps[knum].inBufferA = input_buffer_A##knum;                                                                                                  \
    kComps[knum].inBufferB = input_buffer_B##knum;                                                                                                  \
    kComps[knum].outBufferA = output_buffer_A##knum;                                                                                                \
    kComps[knum].outBufferB = output_buffer_B##knum;                                                                                                \
    kComps[knum].tableBuffer = table_buffer_##knum;

#define SETUP_TEN_KERNELS \
    SETUP_KERNEL(0);      \
    SETUP_KERNEL(1);      \
    SETUP_KERNEL(2);      \
    SETUP_KERNEL(3);      \
    SETUP_KERNEL(4);      \
    SETUP_KERNEL(5);      \
    SETUP_KERNEL(6);      \
    SETUP_KERNEL(7);      \
    SETUP_KERNEL(8);      \
    SETUP_KERNEL(9);

#define PROGRAM_KERNEL(knum)                                                   \
    kernelConfigString = "calculate_crc:{CRC_" + std::to_string(knum) + "}\0"; \
    charBuf = std::strcpy(charBuf, kernelConfigString.c_str());                \
    OCL_CHECK(err, kComps[knum].kernel = cl::Kernel(program, charBuf, &err));

#define PROGRAM_TEN_KERNELS \
    PROGRAM_KERNEL(0);      \
    PROGRAM_KERNEL(1);      \
    PROGRAM_KERNEL(2);      \
    PROGRAM_KERNEL(3);      \
    PROGRAM_KERNEL(4);      \
    PROGRAM_KERNEL(5);      \
    PROGRAM_KERNEL(6);      \
    PROGRAM_KERNEL(7);      \
    PROGRAM_KERNEL(8);      \
    PROGRAM_KERNEL(9);

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

uint32_t *generateStandardCRCTable(uint32_t polynomial, int width)
{
    uint32_t *crcTable = new uint32_t[256];
    for (uint32_t i = 0; i < 256; ++i)
    {
        uint32_t crc = i;
        for (uint32_t j = 0; j < 8; ++j)
        {
            if (crc & 1)
            {
                crc = (crc >> 1) ^ polynomial;
            }
            else
            {
                crc = crc >> 1;
            }
        }
        crcTable[i] = maskCRC(crc, width);
    }
    return crcTable;
}

uint32_t standardCompute(std::vector<unsigned char, aligned_allocator<unsigned char>> data, uint32_t *crcTable, uint32_t crc_initial, uint32_t crc_final, int width)
{
    uint32_t crc = maskCRC(crc_initial, width);
    for (uint8_t byte : data)
    {
        crc = (crc >> 8) ^ crcTable[(crc ^ byte) & 0xFF];
        crc = maskCRC(crc, width);
    }
    return maskCRC(crc ^ crc_final, width);
}

std::vector<uint32_t> generateParallelCRCTables(uint32_t *standardTable, int width, bool reflectInput)
{
    std::vector<uint32_t> crcTables(16 * 256, 0);

    for (int tableIndex = 0; tableIndex < 16; ++tableIndex)
    {
        for (int byteValue = 0; byteValue < 256; ++byteValue)
        {
            std::vector<unsigned char, aligned_allocator<unsigned char>> data(16, 0);
            data[tableIndex] = static_cast<uint8_t>(byteValue & 0xFF);
            if (reflectInput)
            {
                data[tableIndex] = reverseBits(data[tableIndex], 8) & 0xFF;
            }
            uint32_t crc = standardCompute(data, standardTable, 0xFFFFFFFF, 0xFFFFFFFF, width);
            crcTables[(256 * (15 - tableIndex)) + byteValue] = maskCRC(crc, width);
        }
    }
    return crcTables;
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

struct KernelConfig
{
    uint32_t polynomial;
    uint32_t init_val;
    uint32_t xor_out;
    bool refInput;
    bool refOutput;
    int crcWidth;
    int chunkSize;
    std::ifstream infile;
    std::ofstream outfile;
    // data size
    size_t dataSize;
};

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

void printData(const std::vector<unsigned char, aligned_allocator<unsigned char>> &data)
{
    for (uint8_t byte : data)
    {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte) << " ";
    }
    std::cout << std::endl;
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
    int numKernels = 0;
    if (configs.size() == 0)
    {
        std::cout << "No Configurations Loaded!" << std::endl;
        return;
    }
    if (configs.size() == 1)
    {
        numKernels = 10;
    }
    if (configs.size() == 2)
    {
        numKernels = 5;
    }
    if (configs.size() == 3)
    {
        numKernels = 3;
    }
    if (configs.size() == 4)
    {
        numKernels = 2;
    }
    if (configs.size() == 5)
    {
        numKernels = 2;
    }
    if (configs.size() >= 6)
    {
        numKernels = 1;
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
    for (int i = 0; i < 10; i++)
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

int getConfigForKernel(int numConfigs, int kernel)
{
    if (numConfigs == 1)
    {
        return 0;
    }
    if (numConfigs == 2)
    {
        if (kernel < 5)
        {
            return 0;
        }
        else
        {
            return 1;
        }
    }
    if (numConfigs == 3)
    {
        if (kernel < 3)
        {
            return 0;
        }
        if (kernel < 6)
        {
            return 1;
        }
        if (kernel < 9)
        {
            return 2;
        }
        return -1;
    }
    if (numConfigs == 4)
    {
        if (kernel < 2)
        {
            return 0;
        }
        if (kernel < 4)
        {
            return 1;
        }
        if (kernel < 6)
        {
            return 2;
        }
        if (kernel < 8)
        {
            return 3;
        }
        return -1;
    }
    if (numConfigs == 5)
    {
        if (kernel < 2)
        {
            return 0;
        }
        if (kernel < 4)
        {
            return 1;
        }
        if (kernel < 6)
        {
            return 2;
        }
        if (kernel < 8)
        {
            return 3;
        }
        return 4;
    }
    if (kernel < numConfigs)
    {
        return kernel;
    }
    return -1;
}

int main(int argc, char **argv)
{

    // Parse command line arguments
    sda::utils::CmdLineParser parser;

    parser.addSwitch("--xclbin_file", "-x", "Binary file string", "");
    parser.addSwitch("--frequency", "-f", "Operating frequency, in MHz", "300");
    parser.addSwitch("--buf_size_mb", "-m", "Test buffer size, in MB", "16");
    parser.addSwitch("--buf_size_kb", "-k", "Test buffer size, in KB", "0");
    parser.parse(argc, argv);

    std::string xclbinFile = parser.value("xclbin_file");

    float frequency = stof(parser.value("frequency"));
    int64_t buf_size_mb = stoi(parser.value("buf_size_mb"));
    int64_t buf_size_kb = stoi(parser.value("buf_size_kb"));

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
    /*
    std::cout << "Manual Test (\"yes\" or \"no\"): ";
    std::string manual;
    std::cin >> manual;
    bool manualTest = (manual == "yes");
    // */
    std::cout << "Buffer Size: " << buf_size_bytes << std::endl;

    int numKernels = 10; // MAX 10
    // Setup Hardware

    KernelComponents kComps[numKernels];

    // ------ Allocate Host Memory ------
    for (int i = 0; i < numKernels; i++)
    {

        // ---- Input Buffers ----
        kComps[i].hostInA = (unsigned char *)malloc(buf_size_bytes);
        kComps[i].hostInB = (unsigned char *)malloc(buf_size_bytes);

        // ---- Output Buffers ----
        kComps[i].hostOutA = (uint32_t *)malloc(buf_size_bytes);
        kComps[i].hostOutB = (uint32_t *)malloc(buf_size_bytes);
    }

    // Program Device
    cl_int err;
    cl::Context context;
    cl::CommandQueue queue0a, queue0b, queue1a, queue1b, queue2a, queue2b, queue3a, queue3b;
    cl::Kernel kernel0, kernel1;

    auto devices = xcl::get_xil_devices();
    auto device = devices[0];

    auto fileBuf = xcl::read_binary_file(xclbinFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;

    // Setup output file to hold throughput data
    std::ofstream throughput_file("throughput.csv", std::ios::app);

    for (unsigned int i = 0; i < devices.size(); i++)
    {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        for (int k = 0; k < numKernels; k++)
        {
            OCL_CHECK(err, kComps[k].queueA = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
            OCL_CHECK(err, kComps[k].queueB = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
        }

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS)
        {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        }
        else
        {
            std::string kernelConfigString;
            char *charBuf = new char[256];
            PROGRAM_TEN_KERNELS;
            std::cout << "Device[" << i << "]: program successful!\n";

            valid_device = true;
            break;
        }
    }
    if (!valid_device)
    {
        std::cerr << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    long global_mem_size;
    devices[0].getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem_size);
    std::cout << "Global Memory Size: " << global_mem_size << std::endl;
    cl_mem_ext_ptr_t HBM_ext = {0};
    // Setup Kernels
    SETUP_TEN_KERNELS;

    // unsigned long counter = 0;

    bool dualKernel = true;
    bool skipFileReadWrite = false;

    std::ofstream output_file("throughput.txt", std::ios::app);
    // numKernels = 3;
    //  FPGA TOP LOOP

    std::vector<KernelConfig *> configs = std::vector<KernelConfig *>();
    std::string rerun = "yes";
    while (rerun == "yes")
    {
        std::string addConfig = "yes";
        while (addConfig == "yes")
        {
            KernelConfig *cfg = new KernelConfig();
            std::cout << "Enter Config File Name: ";
            std::string filename;
            std::cin >> filename;
            loadConfig(filename, *cfg);
            if (configs.size() >= 10)
            {
                configs.erase(configs.begin());
            }
            configs.push_back(cfg);

            std::cout << "Add another config? (yes/no): ";
            std::cin >> addConfig;
        }
        printConfiguration(configs);
        long loopCount = 0;
        size_t dataProcessed = 0;

        for (int i = 0; i < configs.size(); i++)
        {
            std::cout << "Enter input file for config " << i << ": ";
            std::string input_filename;
            std::cin >> input_filename;
            configs[i]->infile = std::ifstream(input_filename, std::ios::binary);
            if (!configs[i]->infile.is_open())
            {
                std::cerr << "ERROR: Could not open file " << input_filename << std::endl;
                return EXIT_FAILURE;
            }

            // get file size
            size_t dataSize = configs[i]->infile.tellg();
            configs[i]->infile.seekg(0, std::ios::end);
            int numChunks = dataSize / configs[i]->chunkSize;
            dataProcessed += dataSize;
            size_t batchSize = configs[i]->chunkSize;
            if (dataSize / batchSize > loopCount)
            {
                loopCount = dataSize / batchSize;
            }
            while (batchSize < buf_size_bytes)
            {
                batchSize += configs[i]->chunkSize;
            }

            configs[i]->dataSize = batchSize - configs[i]->chunkSize;
            configs[i]->outfile = std::ofstream("output_" + std::to_string(i) + ".bin", std::ios::binary);
        }

        std::cout << "Skip File Read/Write? (yes/no): ";
        std::string skipFile;
        std::cin >> skipFile;
        skipFileReadWrite = (skipFile == "yes");

        // Open Output File

        std::ofstream output_file("output.bin", std::ios::binary);
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < numKernels; k++)
        {
            if (getConfigForKernel(configs.size(), k) == -1)
            {
                continue;
            }
            int configIndex = getConfigForKernel(configs.size(), k);
            load_data_chunk(kComps[k].hostInA, configs[configIndex]->infile, configs[configIndex]->dataSize);
            load_data_chunk(kComps[k].hostInB, configs[configIndex]->infile, configs[configIndex]->dataSize);
        }

        // std::cout << hostIn0A[0] << std::endl;

        bool oddRunCount = false;
        if (loopCount % 2 != 0)
        {
            loopCount -= 1;
            oddRunCount = true;
        }

        // Write the lookup table to the device
        for (int k = 0; k < numKernels; k++)
        {
            if (getConfigForKernel(configs.size(), k) == -1)
            {
                continue;
            }
            int configIndex = getConfigForKernel(configs.size(), k);
            uint32_t revPoly = reverseBits(configs[configIndex]->polynomial, configs[configIndex]->crcWidth);
            std::vector<uint32_t> lookup_table = generateParallelCRCTables(generateStandardCRCTable(revPoly, configs[configIndex]->crcWidth), configs[configIndex]->crcWidth, configs[configIndex]->refInput);
            OCL_CHECK(err, err = kComps[k].queueA.enqueueWriteBuffer(kComps[k].tableBuffer, CL_TRUE, 0, 256 * 16 * sizeof(uint32_t), lookup_table.data(), nullptr, nullptr));
            OCL_CHECK(err, err = kComps[k].queueA.finish());
        }

        bool firstLoop = false;
        // long loopcount = 0;

        while (loopCount > 0)
        {
            loopCount -= 2 * numKernels;

            for (int k = 0; k < numKernels; k++)
            {
                if (getConfigForKernel(configs.size(), k) == -1)
                {
                    continue;
                }
                int configIndex = getConfigForKernel(configs.size(), k);
                uint32_t init_val = configs[configIndex]->init_val;
                int crcWidth = configs[configIndex]->crcWidth;
                int chunkSize = configs[configIndex]->chunkSize;
                int numChunks = configs[configIndex]->dataSize / chunkSize;
                OCL_CHECK(err, err = kComps[k].kernel.setArg(0, kComps[k].inBufferA));                    // Input buffer
                OCL_CHECK(err, err = kComps[k].kernel.setArg(1, kComps[k].outBufferA));                   // Output buffer
                OCL_CHECK(err, err = kComps[k].kernel.setArg(2, kComps[k].tableBuffer));                  // Tables
                OCL_CHECK(err, err = kComps[k].kernel.setArg(3, (static_cast<unsigned int>(numChunks)))); // Number of Chunks
                OCL_CHECK(err, err = kComps[k].kernel.setArg(4, (static_cast<unsigned int>(chunkSize)))); // Chunk Size (in blocks)
                OCL_CHECK(err, err = kComps[k].kernel.setArg(5, static_cast<uint32_t>(crcWidth)));        // crc_size
                OCL_CHECK(err, err = kComps[k].kernel.setArg(6, static_cast<uint32_t>(init_val)));        // init_value

                // Q1: 1. H1in >> B1in // Copy Batch 1 from Host onto FPGA memory
                OCL_CHECK(err, err = kComps[k].queueA.enqueueWriteBuffer(kComps[k].inBufferA, CL_TRUE, 0, buf_size_bytes, kComps[k].hostInA, nullptr, nullptr));

                // Q1: 2. >> K1 >> // Run kernel on first batch
                OCL_CHECK(err, err = kComps[k].queueA.enqueueTask(kComps[k].kernel));

                // Q2: 1. H2in >> B2in // Copy Batch 2 from Host onto FPGA memory
                OCL_CHECK(err, err = kComps[k].queueB.enqueueWriteBuffer(kComps[k].inBufferB, CL_TRUE, 0, buf_size_bytes, kComps[k].hostInB, nullptr, nullptr));
            }

            if (!firstLoop)
            {
                // Save Output from previous batches

                if (!skipFileReadWrite)
                {

                    for (int k = 0; k < numKernels; k++)
                    {
                        if (getConfigForKernel(configs.size(), k) == -1)
                        {
                            continue;
                        }
                        int configIndex = getConfigForKernel(configs.size(), k);
                        int chunkSize = configs[configIndex]->chunkSize;
                        int numChunks = configs[configIndex]->dataSize / chunkSize;
                        int crcWidth = configs[configIndex]->crcWidth;

                        save_to_file(kComps[k].hostOutA, configs[configIndex]->outfile, numChunks, configs[configIndex]->refOutput, crcWidth);
                        save_to_file(kComps[k].hostOutB, configs[configIndex]->outfile, numChunks, configs[configIndex]->refOutput, crcWidth);
                    }
                }
            }
            else
            {
                firstLoop = false;
            }

            // Sync with device

            for (int k = 0; k < numKernels; k++)
            {

                OCL_CHECK(err, err = kComps[k].queueA.finish());
                OCL_CHECK(err, err = kComps[k].queueB.finish());
            }

            // Set Kernel arguments for Queue 2 kernel0
            // Kernel Signature: calculate_crc (void* data_in, void* crc_out, void* tables, unsigned int numChunks, unsigned int chunkSize, unsigned int crc_size, unsigned int init_value)
            for (int k = 0; k < numKernels; k++)
            {
                if (getConfigForKernel(configs.size(), k) == -1)
                {
                    continue;
                }
                int configIndex = getConfigForKernel(configs.size(), k);
                uint32_t init_val = configs[configIndex]->init_val;
                int crcWidth = configs[configIndex]->crcWidth;
                int chunkSize = configs[configIndex]->chunkSize;
                int numChunks = configs[configIndex]->dataSize / chunkSize;

                OCL_CHECK(err, err = kComps[k].kernel.setArg(0, kComps[k].inBufferB));                    // Input buffer
                OCL_CHECK(err, err = kComps[k].kernel.setArg(1, kComps[k].outBufferB));                   // Output buffer
                OCL_CHECK(err, err = kComps[k].kernel.setArg(2, kComps[k].tableBuffer));                  // Tables
                OCL_CHECK(err, err = kComps[k].kernel.setArg(3, (static_cast<unsigned int>(numChunks)))); // Number of Chunks
                OCL_CHECK(err, err = kComps[k].kernel.setArg(4, (static_cast<unsigned int>(chunkSize)))); // Chunk Size (in blocks)
                OCL_CHECK(err, err = kComps[k].kernel.setArg(5, static_cast<uint32_t>(crcWidth)));        // crc_size
                OCL_CHECK(err, err = kComps[k].kernel.setArg(6, static_cast<uint32_t>(init_val)));        // init_value

                // Q1: B1Out >> H1Out
                OCL_CHECK(err, err = kComps[k].queueA.enqueueReadBuffer(kComps[k].outBufferA, CL_TRUE, 0, numChunks * sizeof(uint32_t), kComps[k].hostOutA));

                // Q2: 1. >> K1 >> // Run kernel on second batch

                OCL_CHECK(err, err = kComps[k].queueB.enqueueTask(kComps[k].kernel));

                // Q1: B1Out >> H1Out
                OCL_CHECK(err, err = kComps[k].queueB.enqueueReadBuffer(kComps[k].outBufferB, CL_TRUE, 0, numChunks * sizeof(uint32_t), kComps[k].hostOutB));
            }

            if (loopCount > 1)
            {
                if (!skipFileReadWrite)
                {
                    for (int k = 0; k < numKernels; k++)
                    {
                        if (getConfigForKernel(configs.size(), k) == -1)
                        {
                            continue;
                        }
                        int configIndex = getConfigForKernel(configs.size(), k);
                        int chunkSize = configs[configIndex]->chunkSize;
                        int numChunks = configs[configIndex]->dataSize / chunkSize;
                        int crcWidth = configs[configIndex]->crcWidth;
                        save_to_file(kComps[k].hostOutA, configs[configIndex]->outfile, numChunks, configs[configIndex]->refOutput, crcWidth);
                        save_to_file(kComps[k].hostOutB, configs[configIndex]->outfile, numChunks, configs[configIndex]->refOutput, crcWidth);
                    }
                }
            }

            // Sync
            for (int k = 0; k < numKernels; k++)
            {
                OCL_CHECK(err, err = kComps[k].queueA.finish());
                OCL_CHECK(err, err = kComps[k].queueB.finish());
            }
        }

        if (!skipFileReadWrite)
        {
            for (int k = 0; k < numKernels; k++)
            {
                if (getConfigForKernel(configs.size(), k) == -1)
                {
                    continue;
                }
                int configIndex = getConfigForKernel(configs.size(), k);
                int chunkSize = configs[configIndex]->chunkSize;
                int numChunks = configs[configIndex]->dataSize / chunkSize;
                int crcWidth = configs[configIndex]->crcWidth;
                save_to_file(kComps[k].hostOutA, configs[configIndex]->outfile, numChunks, configs[configIndex]->refOutput, crcWidth);
                save_to_file(kComps[k].hostOutB, configs[configIndex]->outfile, numChunks, configs[configIndex]->refOutput, crcWidth);
            }
        }
        /*
        if (oddRunCount) {
            load_data_chunk(hostIn0B, input_file, dataSize, refInput);
            OCL_CHECK(err, err = queue0b.enqueueWriteBuffer(input_buffer_0B, CL_TRUE, 0, buf_size_bytes, hostIn0B, nullptr, nullptr));
            OCL_CHECK(err, err = queue0b.enqueueTask(kernel1));
            OCL_CHECK(err, err = queue0b.enqueueReadBuffer(output_buffer_0B, CL_TRUE, 0, numChunks * sizeof(uint32_t), hostOut0B));
            save_to_file(hostOut0B, output_file, numChunks, refOutput, crcWidth);
        } // */
        auto stop_time = std::chrono::high_resolution_clock::now();
        auto calc_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);

        std::cout << "Total Compute time: " << std::dec << calc_time.count() << "ms" << std::endl;
        double throughput = ((double)dataProcessed / 1000000.0) * (1000.0 / (double)calc_time.count());
        throughput *= 8.0;
        output_file << "" << throughput << "" << std::endl;
        if (throughput > 1000.0)
        {
            throughput /= 1000.0;
            std::cout << "Throughput: " << throughput << "Gbps " << std::endl;
        }
        else
        {
            std::cout << "Throughput: " << throughput << "Mbps " << std::endl;
        }

        // CHANGE: Add Loop logic to calculate more packets
        rerun = "no";
        for (int i = 0; i < configs.size(); i++)
        {
            configs[i]->infile.close();
            configs[i]->outfile.close();
        }
        std::cout << "Rerun? (yes/no): ";
        std::cin >> rerun;
    }

    for (int i = 0; i < numKernels; i++)
    {
        free(kComps[i].hostInA);
        free(kComps[i].hostInB);
        free(kComps[i].hostOutA);
        free(kComps[i].hostOutB);
    }

    return 0;
}

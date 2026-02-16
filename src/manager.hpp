#ifndef MANAGER_HPP
#define MANAGER_HPP

#include <CL/cl.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "helpers/crc.h"
#include "xcl2.hpp"

struct KernelConfig
{
    uint32_t polynomial;
    uint32_t init_val;
    uint32_t xor_out;
    bool refInput;
    bool refOutput;
    int crcWidth;
    int chunkSize;
    size_t dataSize;
    uint32_t checkMode = 0;
    uint32_t hashAlgorithm = 0;
};

enum KernelCheckMode : uint32_t
{
    CHECK_MODE_CRC = 0,
    CHECK_MODE_TCP_CHECKSUM = 1,
    CHECK_MODE_HASH = 2
};

struct CrcTask
{
    std::vector<unsigned char> data;
    KernelConfig config;
    std::promise<std::vector<uint32_t>> promise;
};

class TaskQueue
{
public:
    void push(CrcTask &&t)
    {
        std::lock_guard<std::mutex> lk(m_);
        q_.push(std::move(t));
        cv_.notify_one();
    }

    bool pop(CrcTask &out)
    {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]
                 { return stop_ || !q_.empty(); });
        if (stop_ && q_.empty())
        {
            return false;
        }

        out = std::move(q_.front());
        q_.pop();
        return true;
    }

    void stop()
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
        cv_.notify_all();
    }

private:
    std::queue<CrcTask> q_;
    std::mutex m_;
    std::condition_variable cv_;
    bool stop_ = false;
};

enum class WorkerKernelType
{
    CRC,
    TCP,
    SHA
};

struct Worker
{
    WorkerKernelType mode = WorkerKernelType::CRC;

    cl::Context context;
    cl::Device device;
    cl::Program program;
    cl::Kernel kernel;
    cl::CommandQueue qH2D;
    cl::CommandQueue qK;
    cl::CommandQueue qD2H;

    size_t buffer_size = 0;
    cl::Buffer dInA;
    cl::Buffer dOutA;

    cl::Buffer **tableBuffers = nullptr;
    size_t tableCount = 0;
    std::vector<KernelConfig> configs;

    std::thread th;
};

class FpgaManager
{
public:
    explicit FpgaManager(const std::string &binaryFile,
                         size_t buffer_size = 16 * 1024 * 1024,
                         int max_cu = 16,
                         int max_workers = -1,
                         std::vector<KernelConfig> configs = {})
    {
        buffer_size_ = buffer_size;

        auto devices = xcl::get_xil_devices();
        if (devices.empty())
            throw std::runtime_error("No devices found");
        device_ = devices[0];
        context_ = cl::Context(device_);

        std::ifstream bin(binaryFile, std::ios::binary);
        if (!bin)
            throw std::runtime_error("Cannot open xclbin: " + binaryFile);
        bin.seekg(0, std::ios::end);
        const size_t nb = static_cast<size_t>(bin.tellg());
        bin.seekg(0, std::ios::beg);
        std::vector<char> buf(nb);
        bin.read(buf.data(), nb);
        cl::Program::Binaries bins{{buf.data(), nb}};
        program_ = cl::Program(context_, {device_}, bins);

        const int crc_cu_available = detect_cu_count("calculate_crc", "CRC", max_cu);
        if (crc_cu_available == 0)
            throw std::runtime_error("No CU instances named CRC_* found in xclbin");

        int crc_workers = crc_cu_available;
        if (max_workers > 0)
            crc_workers = std::min(crc_workers, max_workers);

        const int tcp_probe_limit = std::min(max_cu, 4);
        const int sha_probe_limit = std::min(max_cu, 4);
        const int tcp_cu_available = detect_cu_count("calculate_tcp_checksum", "TCP", tcp_probe_limit);
        const int sha_cu_available = detect_cu_count("calculate_sha256", "SHA", sha_probe_limit);

        int tcp_workers = tcp_cu_available;
        int sha_workers = sha_cu_available;
        if (max_workers > 0)
        {
            tcp_workers = std::min(tcp_workers, max_workers);
            sha_workers = std::min(sha_workers, max_workers);
        }

        std::cout << "Found CRC CU instances: " << crc_cu_available << std::endl;
        std::cout << "Found TCP CU instances: " << tcp_cu_available << std::endl;
        std::cout << "Found SHA CU instances: " << sha_cu_available << std::endl;

        std::vector<KernelConfig> crc_configs;
        for (const auto &cfg : configs)
        {
            if (cfg.checkMode == CHECK_MODE_CRC)
            {
                crc_configs.push_back(cfg);
            }
        }

        crc_workers_.resize(static_cast<size_t>(crc_workers));
        for (int i = 0; i < crc_workers; ++i)
        {
            init_worker(crc_workers_[static_cast<size_t>(i)],
                        WorkerKernelType::CRC,
                        "calculate_crc:{CRC_" + std::to_string(i) + "}",
                        buffer_size,
                        crc_configs);
        }

        tcp_workers_.resize(static_cast<size_t>(tcp_workers));
        for (int i = 0; i < tcp_workers; ++i)
        {
            init_worker(tcp_workers_[static_cast<size_t>(i)],
                        WorkerKernelType::TCP,
                        "calculate_tcp_checksum:{TCP_" + std::to_string(i) + "}",
                        buffer_size,
                        {});
        }

        sha_workers_.resize(static_cast<size_t>(sha_workers));
        for (int i = 0; i < sha_workers; ++i)
        {
            init_worker(sha_workers_[static_cast<size_t>(i)],
                        WorkerKernelType::SHA,
                        "calculate_sha256:{SHA_" + std::to_string(i) + "}",
                        buffer_size,
                        {});
        }

        for (size_t i = 0; i < crc_workers_.size(); ++i)
        {
            crc_workers_[i].th = std::thread([this, i]
                                             { worker_loop(crc_queue_, crc_workers_[i]); });
        }
        for (size_t i = 0; i < tcp_workers_.size(); ++i)
        {
            tcp_workers_[i].th = std::thread([this, i]
                                             { worker_loop(tcp_queue_, tcp_workers_[i]); });
        }
        for (size_t i = 0; i < sha_workers_.size(); ++i)
        {
            sha_workers_[i].th = std::thread([this, i]
                                             { worker_loop(sha_queue_, sha_workers_[i]); });
        }
    }

    ~FpgaManager()
    {
        crc_queue_.stop();
        tcp_queue_.stop();
        sha_queue_.stop();

        join_workers(crc_workers_);
        join_workers(tcp_workers_);
        join_workers(sha_workers_);
    }

    std::future<std::vector<uint32_t>> submit(const std::vector<unsigned char> &data,
                                              const KernelConfig &config,
                                              bool large_split = false)
    {
        const TaskRoute route = route_for_mode(config.checkMode);

        const size_t totalBytes = data.size();
        const size_t chunkBytes = static_cast<size_t>(config.chunkSize);
        if (chunkBytes == 0)
            throw std::runtime_error("chunkSize must be > 0");
        if ((totalBytes % chunkBytes) != 0)
            throw std::runtime_error("Input size must be an integer multiple of chunkSize");

        size_t chunks_per_buf = buffer_size_ / chunkBytes;
        if (chunks_per_buf == 0)
            throw std::runtime_error("chunkSize is larger than worker buffer size");

        size_t data_per_task = chunkBytes * chunks_per_buf;
        if (route.worker_count * buffer_size_ > totalBytes)
        {
            size_t num_chunks = totalBytes / chunkBytes;
            size_t chunks_per_task = num_chunks / route.worker_count;
            chunks_per_task = std::max<size_t>(1, chunks_per_task);
            data_per_task = chunks_per_task * chunkBytes;
        }
        if (!large_split)
        {
            data_per_task = chunkBytes;
        }
        data_per_task = std::max(chunkBytes, data_per_task);

        std::vector<std::future<std::vector<uint32_t>>> futures;
        for (size_t offset = 0; offset < totalBytes; offset += data_per_task)
        {
            const size_t bytesToProcess = std::min(data_per_task, totalBytes - offset);
            CrcTask t;
            t.data = std::vector<unsigned char>(data.begin() + offset, data.begin() + offset + bytesToProcess);
            t.config = config;
            std::promise<std::vector<uint32_t>> p;
            auto fut = p.get_future();
            t.promise = std::move(p);
            route.queue->push(std::move(t));
            futures.push_back(std::move(fut));
        }

        std::cout << "Tasks on queue: " << futures.size() << std::endl;
        return std::async(std::launch::deferred, [futures = std::move(futures)]() mutable
                          {
            std::vector<uint32_t> result;
            for (auto &fut : futures)
            {
                auto part = fut.get();
                result.insert(result.end(), part.begin(), part.end());
            }
            return result; });
    }

    std::vector<uint32_t> calculate_crc(const std::vector<unsigned char> &data,
                                        const KernelConfig &config,
                                        bool large_split = false)
    {
        return submit(data, config, large_split).get();
    }

private:
    struct TaskRoute
    {
        TaskQueue *queue;
        size_t worker_count;
    };

    TaskRoute route_for_mode(uint32_t mode)
    {
        if (mode == CHECK_MODE_CRC)
        {
            if (crc_workers_.empty())
                throw std::runtime_error("No CRC workers available in xclbin");
            return {&crc_queue_, crc_workers_.size()};
        }
        if (mode == CHECK_MODE_TCP_CHECKSUM)
        {
            if (tcp_workers_.empty())
                throw std::runtime_error("No TCP workers available in xclbin");
            return {&tcp_queue_, tcp_workers_.size()};
        }
        if (mode == CHECK_MODE_HASH)
        {
            if (sha_workers_.empty())
                throw std::runtime_error("No SHA workers available in xclbin");
            return {&sha_queue_, sha_workers_.size()};
        }
        throw std::runtime_error("Unsupported checkMode value");
    }

    static void join_workers(std::vector<Worker> &workers)
    {
        for (auto &w : workers)
        {
            if (w.th.joinable())
            {
                w.th.join();
            }
        }
    }

    int detect_cu_count(const std::string &kernel_name,
                        const std::string &instance_prefix,
                        int max_cu)
    {
        int cu_count = 0;
        while (cu_count < max_cu)
        {
            std::string kname = kernel_name + ":{" + instance_prefix + "_" + std::to_string(cu_count) + "}";
            cl_int err = CL_SUCCESS;
            cl::Kernel testK(program_, kname.c_str(), &err);
            if (err != CL_SUCCESS)
            {
                break;
            }
            cu_count++;
        }
        return cu_count;
    }

    static std::vector<uint32_t> execute_crc(Worker &w,
                                             const std::vector<unsigned char> &data,
                                             const KernelConfig &cfg)
    {
        if (cfg.checkMode != CHECK_MODE_CRC)
            throw std::runtime_error("CRC worker received non-CRC task");

        cl_int err = CL_SUCCESS;
        const size_t totalBytes = data.size();
        const size_t chunkBytes = static_cast<size_t>(cfg.chunkSize);
        if (chunkBytes == 0)
            throw std::runtime_error("chunkSize must be > 0");
        if ((totalBytes % chunkBytes) != 0)
            throw std::runtime_error("Input size must be an integer multiple of chunkSize");

        const size_t nChunks = totalBytes / chunkBytes;
        const size_t wordsPerChunk = 1;
        const size_t outputBytesPerChunk = wordsPerChunk * sizeof(uint32_t);
        const size_t chunksPerBuf = std::min(w.buffer_size / chunkBytes, w.buffer_size / outputBytesPerChunk);
        if (chunksPerBuf == 0)
            throw std::runtime_error("buffer_size is too small for configured chunk input/output footprint");

        CRC_Config crc_cfg;
        crc_cfg.polynomial = cfg.polynomial;
        crc_cfg.initial_value = cfg.init_val;
        crc_cfg.final_xor_value = cfg.xor_out;
        crc_cfg.reflect_input = cfg.refInput;
        crc_cfg.reflect_output = cfg.refOutput;
        crc_cfg.width = static_cast<uint8_t>(cfg.crcWidth);
        crc_cfg.chunk_size = static_cast<size_t>(cfg.chunkSize);

        cl::Buffer table_buf;
        int config_index = -1;
        for (size_t i = 0; i < w.configs.size(); ++i)
        {
            const auto &c = w.configs[i];
            if (c.polynomial == cfg.polynomial &&
                c.init_val == cfg.init_val &&
                c.xor_out == cfg.xor_out &&
                c.refInput == cfg.refInput &&
                c.refOutput == cfg.refOutput &&
                c.crcWidth == cfg.crcWidth)
            {
                config_index = static_cast<int>(i);
                table_buf = *w.tableBuffers[static_cast<size_t>(config_index)];
                break;
            }
        }

        if (config_index == -1)
        {
            config_index = static_cast<int>(w.configs.size());
            table_buf = *w.tableBuffers[static_cast<size_t>(config_index)];

            std::vector<uint32_t> flatTbl(16 * 256);
            auto parTbl = create_parallel_tables(crc_cfg);
            for (int i = 0; i < 16; ++i)
            {
                for (int j = 0; j < 256; ++j)
                {
                    flatTbl[(i << 8) + j] = parTbl[i][j];
                }
            }

            OCL_CHECK(err, err = w.qH2D.enqueueWriteBuffer(table_buf,
                                                           CL_TRUE,
                                                           0,
                                                           flatTbl.size() * sizeof(uint32_t),
                                                           flatTbl.data()));
        }

        std::vector<uint8_t> data_ptr(data.begin(), data.end());
        if (!crc_cfg.reflect_input)
        {
            for (size_t i = 0; i < totalBytes; i++)
            {
                if (i % 16 < 4)
                {
                    data_ptr[i] = static_cast<uint8_t>(reflect(data_ptr[i], 8) & 0xFF);
                }
            }
        }

        std::vector<uint32_t> result;
        result.reserve(nChunks * wordsPerChunk);

        for (size_t k = 0; k < nChunks; k += chunksPerBuf)
        {
            const size_t offset = k * chunkBytes;
            const size_t bytesToProcess = std::min(chunksPerBuf * chunkBytes, totalBytes - offset);
            const size_t chunksToProcess = bytesToProcess / chunkBytes;

            std::vector<unsigned char, aligned_allocator<unsigned char>> chunkData(bytesToProcess, 0);
            std::memcpy(chunkData.data(), data_ptr.data() + offset, bytesToProcess);

            OCL_CHECK(err, err = w.kernel.setArg(0, w.dInA));
            OCL_CHECK(err, err = w.kernel.setArg(1, w.dOutA));
            OCL_CHECK(err, err = w.kernel.setArg(2, *w.tableBuffers[static_cast<size_t>(config_index)]));
            OCL_CHECK(err, err = w.kernel.setArg(3, static_cast<uint32_t>(chunksToProcess)));
            OCL_CHECK(err, err = w.kernel.setArg(4, static_cast<uint32_t>(cfg.chunkSize)));
            OCL_CHECK(err, err = w.kernel.setArg(5, static_cast<uint32_t>(cfg.crcWidth)));
            OCL_CHECK(err, err = w.kernel.setArg(6, static_cast<uint32_t>(cfg.init_val)));

            OCL_CHECK(err, err = w.qH2D.enqueueWriteBuffer(w.dInA,
                                                           CL_TRUE,
                                                           0,
                                                           bytesToProcess,
                                                           chunkData.data()));

            cl::NDRange one(1);
            OCL_CHECK(err, err = w.qK.enqueueNDRangeKernel(w.kernel,
                                                           cl::NullRange,
                                                           one,
                                                           one));
            OCL_CHECK(err, err = w.qK.finish());

            std::vector<uint32_t, aligned_allocator<uint32_t>> crcOut(chunksToProcess * wordsPerChunk);
            OCL_CHECK(err, err = w.qD2H.enqueueReadBuffer(w.dOutA,
                                                          CL_TRUE,
                                                          0,
                                                          sizeof(uint32_t) * chunksToProcess * wordsPerChunk,
                                                          crcOut.data()));

            result.insert(result.end(), crcOut.begin(), crcOut.end());
        }

        for (size_t i = 0; i < result.size(); ++i)
        {
            result[i] ^= (cfg.xor_out);
        }
        if (crc_cfg.reflect_output != crc_cfg.reflect_input)
        {
            for (size_t i = 0; i < result.size(); ++i)
            {
                result[i] = reflect(result[i], static_cast<uint8_t>(cfg.crcWidth));
            }
        }

        return result;
    }

    static std::vector<uint32_t> execute_tcp(Worker &w,
                                             const std::vector<unsigned char> &data,
                                             const KernelConfig &cfg)
    {
        if (cfg.checkMode != CHECK_MODE_TCP_CHECKSUM)
            throw std::runtime_error("TCP worker received non-TCP task");

        cl_int err = CL_SUCCESS;
        const size_t totalBytes = data.size();
        const size_t chunkBytes = static_cast<size_t>(cfg.chunkSize);
        if (chunkBytes == 0)
            throw std::runtime_error("chunkSize must be > 0");
        if ((totalBytes % chunkBytes) != 0)
            throw std::runtime_error("Input size must be an integer multiple of chunkSize");

        const size_t nChunks = totalBytes / chunkBytes;
        const size_t wordsPerChunk = 1;
        const size_t outputBytesPerChunk = wordsPerChunk * sizeof(uint32_t);
        const size_t chunksPerBuf = std::min(w.buffer_size / chunkBytes, w.buffer_size / outputBytesPerChunk);
        if (chunksPerBuf == 0)
            throw std::runtime_error("buffer_size is too small for configured chunk input/output footprint");

        std::vector<uint8_t> data_ptr(data.begin(), data.end());
        std::vector<uint32_t> result;
        result.reserve(nChunks * wordsPerChunk);

        for (size_t k = 0; k < nChunks; k += chunksPerBuf)
        {
            const size_t offset = k * chunkBytes;
            const size_t bytesToProcess = std::min(chunksPerBuf * chunkBytes, totalBytes - offset);
            const size_t chunksToProcess = bytesToProcess / chunkBytes;

            std::vector<unsigned char, aligned_allocator<unsigned char>> chunkData(bytesToProcess, 0);
            std::memcpy(chunkData.data(), data_ptr.data() + offset, bytesToProcess);

            OCL_CHECK(err, err = w.kernel.setArg(0, w.dInA));
            OCL_CHECK(err, err = w.kernel.setArg(1, w.dOutA));
            OCL_CHECK(err, err = w.kernel.setArg(2, static_cast<uint32_t>(chunksToProcess)));
            OCL_CHECK(err, err = w.kernel.setArg(3, static_cast<uint32_t>(cfg.chunkSize)));

            OCL_CHECK(err, err = w.qH2D.enqueueWriteBuffer(w.dInA,
                                                           CL_TRUE,
                                                           0,
                                                           bytesToProcess,
                                                           chunkData.data()));

            cl::NDRange one(1);
            OCL_CHECK(err, err = w.qK.enqueueNDRangeKernel(w.kernel,
                                                           cl::NullRange,
                                                           one,
                                                           one));
            OCL_CHECK(err, err = w.qK.finish());

            std::vector<uint32_t, aligned_allocator<uint32_t>> out(chunksToProcess * wordsPerChunk);
            OCL_CHECK(err, err = w.qD2H.enqueueReadBuffer(w.dOutA,
                                                          CL_TRUE,
                                                          0,
                                                          sizeof(uint32_t) * chunksToProcess * wordsPerChunk,
                                                          out.data()));

            result.insert(result.end(), out.begin(), out.end());
        }

        return result;
    }

    static std::vector<uint32_t> execute_sha(Worker &w,
                                             const std::vector<unsigned char> &data,
                                             const KernelConfig &cfg)
    {
        if (cfg.checkMode != CHECK_MODE_HASH)
            throw std::runtime_error("SHA worker received non-SHA task");
        if (cfg.hashAlgorithm != 0)
            throw std::runtime_error("Only SHA-256 hashAlgorithm=0 is implemented in the kernel.");

        cl_int err = CL_SUCCESS;
        const size_t totalBytes = data.size();
        const size_t chunkBytes = static_cast<size_t>(cfg.chunkSize);
        if (chunkBytes == 0)
            throw std::runtime_error("chunkSize must be > 0");
        if ((totalBytes % chunkBytes) != 0)
            throw std::runtime_error("Input size must be an integer multiple of chunkSize");

        const size_t nChunks = totalBytes / chunkBytes;
        const size_t wordsPerChunk = 8;
        const size_t outputBytesPerChunk = wordsPerChunk * sizeof(uint32_t);
        const size_t chunksPerBuf = std::min(w.buffer_size / chunkBytes, w.buffer_size / outputBytesPerChunk);
        if (chunksPerBuf == 0)
            throw std::runtime_error("buffer_size is too small for configured chunk input/output footprint");

        std::vector<uint8_t> data_ptr(data.begin(), data.end());
        std::vector<uint32_t> result;
        result.reserve(nChunks * wordsPerChunk);

        for (size_t k = 0; k < nChunks; k += chunksPerBuf)
        {
            const size_t offset = k * chunkBytes;
            const size_t bytesToProcess = std::min(chunksPerBuf * chunkBytes, totalBytes - offset);
            const size_t chunksToProcess = bytesToProcess / chunkBytes;

            std::vector<unsigned char, aligned_allocator<unsigned char>> chunkData(bytesToProcess, 0);
            std::memcpy(chunkData.data(), data_ptr.data() + offset, bytesToProcess);

            OCL_CHECK(err, err = w.kernel.setArg(0, w.dInA));
            OCL_CHECK(err, err = w.kernel.setArg(1, w.dOutA));
            OCL_CHECK(err, err = w.kernel.setArg(2, static_cast<uint32_t>(chunksToProcess)));
            OCL_CHECK(err, err = w.kernel.setArg(3, static_cast<uint32_t>(cfg.chunkSize)));

            OCL_CHECK(err, err = w.qH2D.enqueueWriteBuffer(w.dInA,
                                                           CL_TRUE,
                                                           0,
                                                           bytesToProcess,
                                                           chunkData.data()));

            cl::NDRange one(1);
            OCL_CHECK(err, err = w.qK.enqueueNDRangeKernel(w.kernel,
                                                           cl::NullRange,
                                                           one,
                                                           one));
            OCL_CHECK(err, err = w.qK.finish());

            std::vector<uint32_t, aligned_allocator<uint32_t>> out(chunksToProcess * wordsPerChunk);
            OCL_CHECK(err, err = w.qD2H.enqueueReadBuffer(w.dOutA,
                                                          CL_TRUE,
                                                          0,
                                                          sizeof(uint32_t) * chunksToProcess * wordsPerChunk,
                                                          out.data()));

            result.insert(result.end(), out.begin(), out.end());
        }

        return result;
    }

    static std::vector<uint32_t> execute_task(Worker &w,
                                              const std::vector<unsigned char> &data,
                                              const KernelConfig &cfg)
    {
        switch (w.mode)
        {
        case WorkerKernelType::CRC:
            return execute_crc(w, data, cfg);
        case WorkerKernelType::TCP:
            return execute_tcp(w, data, cfg);
        case WorkerKernelType::SHA:
            return execute_sha(w, data, cfg);
        }
        throw std::runtime_error("Unsupported worker mode");
    }

    static void worker_loop(TaskQueue &queue, Worker &w)
    {
        CrcTask task;
        while (queue.pop(task))
        {
            try
            {
                auto out = execute_task(w, task.data, task.config);
                task.promise.set_value(std::move(out));
            }
            catch (...)
            {
                try
                {
                    task.promise.set_exception(std::current_exception());
                }
                catch (...)
                {
                }
            }
        }
    }

    void init_worker(Worker &w,
                     WorkerKernelType mode,
                     const std::string &kernel_name,
                     size_t buffer_size,
                     const std::vector<KernelConfig> &configs)
    {
        w.mode = mode;
        w.context = context_;
        w.device = device_;
        w.program = program_;
        w.buffer_size = buffer_size;
        w.configs = configs;

        cl_int err = CL_SUCCESS;
        w.kernel = cl::Kernel(w.program, kernel_name.c_str(), &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("Kernel create failed: " + kernel_name);

        w.qH2D = cl::CommandQueue(w.context, w.device, 0, &err);
        w.qK = cl::CommandQueue(w.context, w.device, 0, &err);
        w.qD2H = cl::CommandQueue(w.context, w.device, 0, &err);

        w.dInA = cl::Buffer(w.context, CL_MEM_READ_ONLY, buffer_size, nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("dInA alloc failed");
        OCL_CHECK(err, err = w.kernel.setArg(0, w.dInA));

        w.dOutA = cl::Buffer(w.context, CL_MEM_WRITE_ONLY, buffer_size, nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("dOutA alloc failed");
        OCL_CHECK(err, err = w.kernel.setArg(1, w.dOutA));

        if (mode != WorkerKernelType::CRC)
        {
            w.tableBuffers = nullptr;
            w.tableCount = 0;
            return;
        }

        const int numTables = static_cast<int>(w.configs.size() + 1);
        w.tableCount = static_cast<size_t>(numTables);
        w.tableBuffers = new cl::Buffer *[w.tableCount];

        for (int i = 0; i < numTables; ++i)
        {
            cl::Buffer *table_buf = new cl::Buffer(w.context,
                                                   CL_MEM_READ_ONLY,
                                                   256 * 16 * sizeof(uint32_t),
                                                   nullptr,
                                                   &err);
            OCL_CHECK(err, err = w.kernel.setArg(2, *table_buf));

            if ((size_t)i < w.configs.size())
            {
                CRC_Config crc_cfg;
                crc_cfg.polynomial = w.configs[(size_t)i].polynomial;
                crc_cfg.initial_value = w.configs[(size_t)i].init_val;
                crc_cfg.final_xor_value = w.configs[(size_t)i].xor_out;
                crc_cfg.reflect_input = w.configs[(size_t)i].refInput;
                crc_cfg.reflect_output = w.configs[(size_t)i].refOutput;
                crc_cfg.width = static_cast<uint8_t>(w.configs[(size_t)i].crcWidth);
                crc_cfg.chunk_size = static_cast<size_t>(w.configs[(size_t)i].chunkSize);

                auto parTbl = create_parallel_tables(crc_cfg);
                std::vector<uint32_t> flatTbl(16 * 256);
                for (int m = 0; m < 16; ++m)
                {
                    for (int n = 0; n < 256; ++n)
                    {
                        flatTbl[(m << 8) + n] = parTbl[m][n];
                    }
                }

                OCL_CHECK(err, err = w.qH2D.enqueueWriteBuffer(*table_buf,
                                                               CL_TRUE,
                                                               0,
                                                               flatTbl.size() * sizeof(uint32_t),
                                                               flatTbl.data()));
            }
            else
            {
                std::vector<uint32_t> flatTbl(16 * 256, 0);
                OCL_CHECK(err, err = w.qH2D.enqueueWriteBuffer(*table_buf,
                                                               CL_TRUE,
                                                               0,
                                                               flatTbl.size() * sizeof(uint32_t),
                                                               flatTbl.data()));
            }

            w.tableBuffers[(size_t)i] = table_buf;
        }

        if (err != CL_SUCCESS)
            throw std::runtime_error("table buffer setup failed");
    }

private:
    cl::Context context_;
    cl::Program program_;
    cl::Device device_;

    size_t buffer_size_ = 0;

    TaskQueue crc_queue_;
    TaskQueue tcp_queue_;
    TaskQueue sha_queue_;

    std::vector<Worker> crc_workers_;
    std::vector<Worker> tcp_workers_;
    std::vector<Worker> sha_workers_;
};

#endif // MANAGER_HPP

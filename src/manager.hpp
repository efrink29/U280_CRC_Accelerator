#ifndef MANAGER_HPP
#define MANAGER_HPP

#include <CL/cl.h>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>

#include "xcl2.hpp"

uint32_t *generateStandardCRCTable(uint32_t polynomial, int width);
std::vector<uint32_t> generateParallelCRCTables(uint32_t *standardTable, int width, bool reflectInput);

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

struct Worker
{

    cl::Context context;
    cl::Device device;
    cl::Program program;
    cl::Kernel kernel;
    cl::CommandQueue qH2D;
    cl::CommandQueue qK;
    cl::CommandQueue qD2H;

    // Device Memory
    size_t buffer_size;
    cl::Buffer dInA, dInB, dOutA, dOutB, dTbl;

    // Host Memory
    std::vector<unsigned char, aligned_allocator<unsigned char>> hInA, hInB;
    std::vector<uint32_t, aligned_allocator<uint32_t>> hOutA, hOutB;

    std::thread th;
};

class FpgaManager
{
public:
    explicit FpgaManager(const std::string &binaryFile, size_t buffer_size = 16 * 1024 * 1024, int max_cu = 14, int max_workers = -1)
    {
        // Initialize Device
        auto devices = xcl::get_xil_devices();
        if (devices.empty())
            throw std::runtime_error("No devices found");
        device_ = devices[0];
        context_ = cl::Context(device_);

        // Load xclbin
        std::ifstream bin(binaryFile, std::ios::binary);
        if (!bin)
            throw std::runtime_error("Cannot open xclbin: " + binaryFile);
        bin.seekg(0, std::ios::end);
        size_t nb = static_cast<size_t>(bin.tellg());
        bin.seekg(0, std::ios::beg);
        std::vector<char> buf(nb);
        bin.read(buf.data(), nb);
        cl::Program::Binaries bins{{buf.data(), nb}};
        program_ = cl::Program(context_, {device_}, bins);

        // Identify availible Kernels
        // std::cout << "Check 1" << std::endl;
        int cu_count = 0;
        while (cu_count < max_cu)
        {
            std::string kname = "calculate_crc:{CRC_" + std::to_string(cu_count) + "}";
            cl_int err = CL_SUCCESS;
            cl::Kernel testK(program_, kname.c_str(), &err);
            if (err != CL_SUCCESS)
            {
                break;
            }
            cu_count++;
        }
        std::cout << "Found " << cu_count << " CU instances" << std::endl;
        if (cu_count == 0)
            throw std::runtime_error("No CU instances named CRC_* found in xclbin");

        if (max_workers > 0)
            cu_count = std::min(cu_count, max_workers);

        // Build workers (one per Kernel)

        workers_.resize(cu_count);
        for (int i = 0; i < cu_count; ++i)
            init_worker(i, buffer_size);

        // Start threads
        for (int i = 0; i < cu_count; ++i)
        {
            workers_[i].th = std::thread([this, i]
                                         { worker_loop(i); });
        }
    }

    ~FpgaManager()
    {
        // Hold till all tasks exit
        queue_.stop();
        for (auto &w : workers_)
            if (w.th.joinable())
                w.th.join();
    }

    // Non-Blocking CRC calculation
    std::future<std::vector<uint32_t>>
    submit(const std::vector<unsigned char> &data, const KernelConfig &config)
    {
        CrcTask t;
        t.data = data;
        t.config = config;
        std::promise<std::vector<uint32_t>> p;
        auto fut = p.get_future();
        t.promise = std::move(p);
        queue_.push(std::move(t));
        return fut;
    }

    // Blocking Syncronous Function
    std::vector<uint32_t>
    calculate_crc(const std::vector<unsigned char> &data, const KernelConfig &config)
    {
        return submit(data, config).get();
    }

private:
    static std::vector<uint32_t> execute_crc(Worker &w,
                                             const std::vector<unsigned char> &data,
                                             const KernelConfig &cfg)
    {
        cl_int err = CL_SUCCESS;

        const size_t totalBytes = data.size();
        const size_t chunkBytes = static_cast<size_t>(cfg.chunkSize);
        if (chunkBytes == 0)
            throw std::runtime_error("chunkSize must be > 0");
        const size_t chunksPerBuf = w.buffer_size / chunkBytes;
        if (chunksPerBuf == 0)
            throw std::runtime_error("buffer_size must be >= chunkSize");

        // Generate Tables (Could be Cached?)
        uint32_t *stdTbl = generateStandardCRCTable(cfg.polynomial, cfg.crcWidth);
        std::vector<uint32_t> parTbl = generateParallelCRCTables(stdTbl, cfg.crcWidth, cfg.refInput);
        delete[] stdTbl;

        if (w.dTbl() == nullptr)
        {
        }
        OCL_CHECK(err, err = w.qH2D.enqueueWriteBuffer(w.dTbl, CL_FALSE, 0,
                                                       parTbl.size() * sizeof(uint32_t), parTbl.data()));
        w.qH2D.finish();

        auto launchForBuf = [&](bool A, size_t valid) -> std::pair<cl::Event, cl::Event>
        {
            const size_t nChunks = valid / chunkBytes;
            if (A)
                std::fill(w.hInA.begin() + valid, w.hInA.end(), 0);
            else
                std::fill(w.hInB.begin() + valid, w.hInB.end(), 0);

            cl::Event evW, evK;
            if (A)
            {
                OCL_CHECK(err, err = w.kernel.setArg(0, w.dInA));
                OCL_CHECK(err, err = w.kernel.setArg(1, w.dOutA));
            }
            else
            {
                OCL_CHECK(err, err = w.kernel.setArg(0, w.dInB));
                OCL_CHECK(err, err = w.kernel.setArg(1, w.dOutB));
            }
            OCL_CHECK(err, err = w.kernel.setArg(2, w.dTbl));
            OCL_CHECK(err, err = w.kernel.setArg(3, static_cast<uint32_t>(nChunks)));
            OCL_CHECK(err, err = w.kernel.setArg(4, static_cast<uint32_t>(chunkBytes)));
            OCL_CHECK(err, err = w.kernel.setArg(5, static_cast<uint32_t>(cfg.crcWidth)));
            OCL_CHECK(err, err = w.kernel.setArg(6, static_cast<uint32_t>(cfg.init_val)));

            if (A)
            {
                OCL_CHECK(err, err = w.qH2D.enqueueWriteBuffer(
                                   w.dInA, CL_FALSE, 0, w.buffer_size, w.hInA.data(),
                                   nullptr, &evW));
            }
            else
            {
                OCL_CHECK(err, err = w.qH2D.enqueueWriteBuffer(
                                   w.dInB, CL_FALSE, 0, w.buffer_size, w.hInB.data(),
                                   nullptr, &evW));
            }

            // Kernel waits on the write
            std::vector<cl::Event> deps{evW};
            OCL_CHECK(err, err = w.qK.enqueueTask(w.kernel, &deps, &evK));
            return {evW, evK};
        };

        auto readBack = [&](bool A, size_t nChunks, const cl::Event &dep) -> cl::Event
        {
            cl::Event evR;

            if (A)
                w.hOutA.resize(nChunks);
            else
                w.hOutB.resize(nChunks);

            std::vector<cl::Event> deps{dep};
            if (A)
            {
                OCL_CHECK(err, err = w.qD2H.enqueueReadBuffer(
                                   w.dOutA, CL_FALSE, 0, nChunks * sizeof(uint32_t),
                                   w.hOutA.data(), &deps, &evR));
            }
            else
            {
                OCL_CHECK(err, err = w.qD2H.enqueueReadBuffer(
                                   w.dOutB, CL_FALSE, 0, nChunks * sizeof(uint32_t),
                                   w.hOutB.data(), &deps, &evR));
            }
            return evR;
        };

        std::vector<uint32_t> result;
        result.reserve((totalBytes + chunkBytes - 1) / chunkBytes);

        size_t off = 0;
        // Prime A
        size_t bytes = std::min(w.buffer_size, totalBytes - off);
        std::copy(data.begin() + off, data.begin() + off + bytes, w.hInA.begin());
        auto [wA, kA] = launchForBuf(true, bytes);
        size_t prodA = bytes / chunkBytes;
        cl::Event rA = (prodA ? readBack(true, prodA, kA) : cl::Event());
        off += bytes;

        // Prime B if needed
        size_t prodB = 0;
        cl::Event rB;
        if (off < totalBytes)
        {
            bytes = std::min(w.buffer_size, totalBytes - off);
            std::copy(data.begin() + off, data.begin() + off + bytes, w.hInB.begin());
            auto [wB, kB] = launchForBuf(false, bytes);
            prodB = bytes / chunkBytes;
            rB = (prodB ? readBack(false, prodB, kB) : cl::Event());
            off += bytes;
        }

        // Read Back Results
        if (prodA)
        {
            rA.wait();
            result.insert(result.end(), w.hOutA.begin(), w.hOutA.begin() + prodA);
        }
        if (prodB)
        {
            rB.wait();
            result.insert(result.end(), w.hOutB.begin(), w.hOutB.begin() + prodB);
        }

        bool useA = true;
        while (off < totalBytes)
        {
            // Write & launch next on A or B
            bool A = useA;
            bytes = std::min(w.buffer_size, totalBytes - off);
            if (A)
            {
                std::copy(data.begin() + off, data.begin() + off + bytes, w.hInA.begin());
                auto [wE, kE] = launchForBuf(true, bytes);
                size_t n = bytes / chunkBytes;
                cl::Event rE = (n ? readBack(true, n, kE) : cl::Event());
                if (n)
                {
                    rE.wait();
                    result.insert(result.end(), w.hOutA.begin(), w.hOutA.begin() + n);
                }
            }
            else
            {
                std::copy(data.begin() + off, data.begin() + off + bytes, w.hInB.begin());
                auto [wE, kE] = launchForBuf(false, bytes);
                size_t n = bytes / chunkBytes;
                cl::Event rE = (n ? readBack(false, n, kE) : cl::Event());
                if (n)
                {
                    rE.wait();
                    result.insert(result.end(), w.hOutB.begin(), w.hOutB.begin() + n);
                }
            }
            off += bytes;
            useA = !useA;
        }

        // Sync all queues
        w.qH2D.finish();
        w.qK.finish();
        w.qD2H.finish();

        return result;
    }

    void worker_loop(int idx)
    {
        auto &w = workers_[idx];
        CrcTask task;
        while (queue_.pop(task))
        {
            try
            {
                auto out = execute_crc(w, task.data, task.config);
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

    void init_worker(int cu_index, size_t buffer_size)
    {
        Worker w;
        w.context = context_;
        w.device = device_;
        w.program = program_;
        w.buffer_size = buffer_size;

        cl_int err = CL_SUCCESS;
        std::string kname = "calculate_crc:{CRC_" + std::to_string(cu_index) + "}";
        w.kernel = cl::Kernel(w.program, kname.c_str(), &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("Kernel create failed");

        w.qH2D = cl::CommandQueue(w.context, w.device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        w.qK = cl::CommandQueue(w.context, w.device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        w.qD2H = cl::CommandQueue(w.context, w.device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);

        // Host Memory
        w.hInA.assign(buffer_size, 0);
        w.hInB.assign(buffer_size, 0);
        w.hOutA.assign(1, 0);
        w.hOutB.assign(1, 0);

        // Device buffers
        w.dInA = cl::Buffer(w.context, CL_MEM_READ_ONLY, buffer_size, nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("dInA alloc failed");
        w.dInB = cl::Buffer(w.context, CL_MEM_READ_ONLY, buffer_size, nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("dInB alloc failed");
        w.dOutA = cl::Buffer(w.context, CL_MEM_WRITE_ONLY, buffer_size * sizeof(uint32_t), nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("dOutA alloc failed");
        w.dOutB = cl::Buffer(w.context, CL_MEM_WRITE_ONLY, buffer_size * sizeof(uint32_t), nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("dOutB alloc failed");
        w.dTbl = cl::Buffer(w.context, CL_MEM_READ_ONLY, 256 * 16 * sizeof(uint32_t), nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("dTbl alloc failed");

        // Bind buffers to CU
        OCL_CHECK(err, err = w.kernel.setArg(0, w.dInA));
        OCL_CHECK(err, err = w.kernel.setArg(1, w.dOutA));
        OCL_CHECK(err, err = w.kernel.setArg(0, w.dInB));
        OCL_CHECK(err, err = w.kernel.setArg(1, w.dOutB));
        OCL_CHECK(err, err = w.kernel.setArg(2, w.dTbl));

        workers_[cu_index] = std::move(w);
    }

private:
    cl::Context context_;
    cl::Program program_;
    cl::Device device_;

    TaskQueue queue_;
    std::vector<Worker> workers_;
};

#endif // MANAGER_HPP

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include "../src/helpers/crc.h"

struct Config {
  uint8_t width = 32;
  uint32_t polynomial = 0x04C11DB7u;
  uint32_t init = 0xFFFFFFFFu;
  bool ref_in = true;
  bool ref_out = true;
  uint32_t xor_out = 0xFFFFFFFFu;
  uint32_t chunk_size = 65536;
  std::string mode = "CRC";
};

static uint32_t parse_u32(const std::string& s)
{
  return static_cast<uint32_t>(std::stoul(s, nullptr, 0));
}

static bool parse_bool01(const std::string& s)
{
  return s == "1" || s == "true" || s == "TRUE";
}

static Config load_config(const std::string& path)
{
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Failed to open config: " + path);

  std::vector<std::string> lines;
  std::string tok;
  while (in >> tok)
    lines.push_back(tok);

  if (lines.size() < 7)
    throw std::runtime_error("Config has fewer than 7 tokens: " + path);

  Config c;
  c.width = static_cast<uint8_t>(std::stoul(lines[0], nullptr, 0));
  c.polynomial = parse_u32(lines[1]);
  c.init = parse_u32(lines[2]);
  c.ref_in = parse_bool01(lines[3]);
  c.ref_out = parse_bool01(lines[4]);
  c.xor_out = parse_u32(lines[5]);
  c.chunk_size = parse_u32(lines[6]);
  if (lines.size() >= 8)
    c.mode = lines[7];
  return c;
}

static std::vector<uint8_t> read_binary(const std::string& path)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in)
    throw std::runtime_error("Failed to open input: " + path);
  auto size = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(size);
  if (size && !in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size)))
    throw std::runtime_error("Failed to read input: " + path);
  return data;
}

int main(int argc, char* argv[])
{
  try {
    if (argc != 4) {
      std::cerr << "Usage: " << argv[0] << " <xclbin> <config_file> <input_bin>\n";
      return 2;
    }

    const std::string xclbin = argv[1];
    const std::string config_file = argv[2];
    const std::string input_file = argv[3];

    Config cfg = load_config(config_file);
    if (cfg.mode != "CRC") {
      std::cerr << "Only CRC mode is supported by this runner. Mode in config: " << cfg.mode << "\n";
      return 3;
    }

    auto data = read_binary(input_file);
    if (cfg.chunk_size == 0)
      throw std::runtime_error("chunk_size must be > 0");
    const size_t tail = data.size() % cfg.chunk_size;
    if (tail)
      data.resize(data.size() - tail);
    if (data.empty())
      throw std::runtime_error("No complete chunks in input after trimming to chunk size");

    const size_t num_chunks = data.size() / cfg.chunk_size;
    const size_t out_words = num_chunks;

    CRC_Config crc_cfg{};
    crc_cfg.polynomial = cfg.polynomial;
    crc_cfg.initial_value = cfg.init;
    crc_cfg.reflect_input = cfg.ref_in;
    crc_cfg.reflect_output = cfg.ref_out;
    crc_cfg.final_xor_value = cfg.xor_out;
    crc_cfg.width = cfg.width;
    crc_cfg.chunk_size = cfg.chunk_size;

    uint32_t** tables = create_parallel_tables(crc_cfg);
    std::vector<uint32_t> flat_tables(16 * 256);
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < 256; ++j)
        flat_tables[(i << 8) + j] = tables[i][j];

    xrt::device device(0);
    auto uuid = device.load_xclbin(xclbin);
    xrt::kernel krnl(device, uuid, "calculate_crc:{CRC_0}");

    xrt::bo in_bo(device, data.size(), krnl.group_id(0));
    xrt::bo out_bo(device, out_words * sizeof(uint32_t), krnl.group_id(1));
    xrt::bo tbl_bo(device, flat_tables.size() * sizeof(uint32_t), krnl.group_id(2));

    in_bo.write(data.data());
    tbl_bo.write(flat_tables.data());
    in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    tbl_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = krnl(in_bo,
                    out_bo,
                    tbl_bo,
                    static_cast<uint32_t>(num_chunks),
                    cfg.chunk_size,
                    static_cast<uint32_t>(cfg.width),
                    cfg.init,
                    0u);
    run.wait();

    std::vector<uint32_t> fpga(out_words, 0);
    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    out_bo.read(fpga.data());

    for (auto& v : fpga)
      v ^= cfg.xor_out;
    if (cfg.ref_out != cfg.ref_in) {
      for (auto& v : fpga)
        v = reflect(v, cfg.width);
    }

    std::vector<uint32_t> host(out_words, 0);
    for (size_t i = 0; i < num_chunks; ++i)
      host[i] = parallel_compute(data.data() + (i * cfg.chunk_size), cfg.chunk_size, crc_cfg);

    const size_t nprint = std::min<size_t>(5, out_words);
    size_t mismatches = 0;
    for (size_t i = 0; i < out_words; ++i)
      mismatches += (host[i] != fpga[i]) ? 1 : 0;

    std::cout << "Chunks: " << out_words << "\n";
    for (size_t i = 0; i < nprint; ++i) {
      std::cout << "Chunk[" << i << "] host=0x" << std::hex << std::setw(8) << std::setfill('0') << host[i]
                << " fpga=0x" << std::setw(8) << fpga[i]
                << (host[i] == fpga[i] ? " [OK]" : " [MISMATCH]") << std::dec << "\n";
    }
    std::cout << "Mismatches: " << mismatches << "\n";

    for (int i = 0; i < 16; ++i)
      delete[] tables[i];
    delete[] tables;

    return mismatches ? 1 : 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}


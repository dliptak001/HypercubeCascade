#include "RamanExtract.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

void LoadExtractor(BaselineExtractor& ex, const std::string& stem)
{
    ex.LoadReadout(stem);
    if (!ex.cascade().readout().IsTrained())
    {
        throw std::runtime_error(
            "LoadExtractor: LoadReadout did not mark trained");
    }
}

void ExtractSplit(BaselineExtractor& ex, const RamanSplit& split,
                  std::span<float> baselines_out)
{
    if (ex.N() != kN || ex.NumOutputs() != kN)
    {
        throw std::invalid_argument(
            "ExtractSplit: extractor N / outputs must equal kN");
    }
    if (split.count == 0)
        throw std::invalid_argument("ExtractSplit: empty split");
    if (baselines_out.size() != split.count * kN)
    {
        throw std::invalid_argument(
            "ExtractSplit: output size must be count * kN");
    }

    for (size_t i = 0; i < split.count; ++i)
    {
        ex.Predict(split.Spectrum(i),
                   {baselines_out.data() + i * kN, kN});
    }
}

void WriteRamanRow(const std::filesystem::path& path,
                   std::span<const float> row)
{
    if (row.size() != kN)
    {
        throw std::invalid_argument(
            "WriteRamanRow: row must have kN values");
    }

    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot open " + path.string());

    char buf[64];
    for (size_t i = 0; i < row.size(); ++i)
    {
        if (i != 0)
            out << ',';
        const int n = std::snprintf(
            buf, sizeof(buf), "%.9g", static_cast<double>(row[i]));
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(buf))
            throw std::runtime_error("WriteRamanRow: format failed");
        out << buf;
    }
    out << '\n';
    if (!out)
        throw std::runtime_error("write failed " + path.string());
}

void WritePredictions(const std::filesystem::path& out_dir,
                      std::span<const int> indices,
                      std::span<const float> baselines_flat,
                      const std::string& stem,
                      const std::filesystem::path& split_dir)
{
    if (indices.empty())
        throw std::invalid_argument("WritePredictions: empty index list");
    if (baselines_flat.size() != indices.size() * kN)
    {
        throw std::invalid_argument(
            "WritePredictions: buffer size must be indices * kN");
    }

    std::filesystem::create_directories(out_dir);

    std::ofstream man(out_dir / "manifest.txt");
    if (!man)
    {
        throw std::runtime_error(
            "cannot open " + (out_dir / "manifest.txt").string());
    }
    man << "stem=" << stem << '\n'
        << "split=" << split_dir.string() << '\n'
        << "indices=";
    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (i != 0)
            man << ',';
        man << indices[i];
    }
    man << '\n';
    if (!man)
        throw std::runtime_error("write failed for manifest.txt");

    for (size_t i = 0; i < indices.size(); ++i)
    {
        WriteRamanRow(out_dir / (std::to_string(indices[i]) + ".pred.txt"),
                      {baselines_flat.data() + i * kN, kN});
    }
}

#include "RamanDataset.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int> ListDataIndices(const std::filesystem::path& dir)
{
    std::vector<int> idx;
    for (const auto& entry : std::filesystem::directory_iterator(dir))
    {
        if (!entry.is_regular_file())
            continue;
        const auto name = entry.path().filename().string();
        static constexpr char kSuffix[] = ".data.txt";
        static constexpr size_t kSuffixLen = sizeof(kSuffix) - 1;
        if (name.size() <= kSuffixLen ||
            name.compare(name.size() - kSuffixLen, kSuffixLen, kSuffix) != 0)
        {
            continue;
        }
        const auto stem = name.substr(0, name.size() - kSuffixLen);
        int v = 0;
        const auto parsed = std::from_chars(
            stem.data(), stem.data() + stem.size(), v);
        if (parsed.ec != std::errc{} || parsed.ptr != stem.data() + stem.size()
            || v < 0)
        {
            continue;
        }
        idx.push_back(v);
    }
    std::sort(idx.begin(), idx.end());
    return idx;
}

void ReadSpectrum(const std::filesystem::path& path, float* dest)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path.string());

    std::string line;
    if (!std::getline(in, line))
        throw std::runtime_error("empty file " + path.string());

    const char* p = line.c_str();
    char* end = nullptr;
    size_t n = 0;
    while (n < kN)
    {
        dest[n] = std::strtof(p, &end);
        if (end == p)
            break;
        ++n;
        if (*end == ',')
        {
            p = end + 1;
            continue;
        }
        break;
    }
    if (n != kN)
    {
        throw std::runtime_error("expected 2048 values in " + path.string()
                                 + ", got " + std::to_string(n));
    }

    const char* tail = end;
    if (*tail == ',')
    {
        throw std::runtime_error("trailing data in " + path.string());
    }
    while (*tail == ' ' || *tail == '\t' || *tail == '\r')
        ++tail;
    if (*tail != '\0')
        throw std::runtime_error("trailing data in " + path.string());
}

} // namespace

RamanSplit LoadRamanSplit(const std::filesystem::path& dir, size_t prefix)
{
    if (!std::filesystem::is_directory(dir))
        throw std::runtime_error("not a directory: " + dir.string());

    const auto indices = ListDataIndices(dir);
    if (indices.empty())
        throw std::runtime_error("no *.data.txt in " + dir.string());

    size_t n = indices.size();
    if (prefix > 0)
    {
        if (prefix > n)
        {
            throw std::runtime_error(
                dir.string() + " has " + std::to_string(n)
                + " spectra, prefix=" + std::to_string(prefix));
        }
        n = prefix;
    }

    RamanSplit split;
    split.count = n;
    split.spectra.resize(n * kN);
    split.baselines.resize(n * kN);

    for (size_t i = 0; i < n; ++i)
    {
        const auto stem = std::to_string(indices[i]);
        ReadSpectrum(dir / (stem + ".data.txt"),
                     split.spectra.data() + i * kN);
        ReadSpectrum(dir / (stem + ".label.txt"),
                     split.baselines.data() + i * kN);
    }
    return split;
}

RamanSplit LoadRamanIndices(const std::filesystem::path& dir,
                            std::span<const int> indices)
{
    if (!std::filesystem::is_directory(dir))
        throw std::runtime_error("not a directory: " + dir.string());
    if (indices.empty())
        throw std::invalid_argument("LoadRamanIndices: empty index list");

    RamanSplit split;
    split.count = indices.size();
    split.spectra.resize(split.count * kN);
    split.baselines.resize(split.count * kN);

    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (indices[i] < 0)
        {
            throw std::invalid_argument(
                "LoadRamanIndices: index must be >= 0");
        }
        const auto stem = std::to_string(indices[i]);
        ReadSpectrum(dir / (stem + ".data.txt"),
                     split.spectra.data() + i * kN);
        ReadSpectrum(dir / (stem + ".label.txt"),
                     split.baselines.data() + i * kN);
    }
    return split;
}

#include "RamanDataset.h"
#include "RamanExtract.h"
#include "RamanPaths.h"

#include <cstdio>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr const char* kSplit = "Validation"; //"Validation";
// Dataset file numbers: Training/0.data.txt, Training/42.data.txt, …
static constexpr int kIndices[] = {1971, 1972, 1973, 1974};
static constexpr const char* kOutDir = "C:/HypercubeCascade/RamanModels/extracted";

int main()
{
    int exit_code = 1;
    try
    {
        const std::span<const int> indices(kIndices);
        const auto split_dir =
            std::filesystem::path(kRamanDataRoot) / kSplit;
        const std::string stem(kRamanModelStem);
        const std::filesystem::path out_dir(kOutDir);

        BaselineExtractor ex(MakeBaseConfig());
        LoadExtractor(ex, stem);
        const auto split = LoadRamanIndices(split_dir, indices);

        std::vector<float> preds(split.count * kN);
        ExtractSplit(ex, split, preds);
        WritePredictions(out_dir, indices, preds, stem, split_dir);

        std::printf("cascade_raman_extract: extracted n=%zu stem=%s -> %s\n",
                    split.count, stem.c_str(), out_dir.string().c_str());
        std::fflush(stdout);
        exit_code = 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "cascade_raman_extract: %s\n", e.what());
    }
    return exit_code;
}

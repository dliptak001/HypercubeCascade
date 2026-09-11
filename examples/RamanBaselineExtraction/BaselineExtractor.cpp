#include "BaselineExtractor.h"
#include "RamanDataset.h"
#include "RamanNorm.h"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <vector>

BaselineExtractor::BaselineExtractor()
    : BaselineExtractor(MakeBaseConfig())
{
}

BaselineExtractor::BaselineExtractor(const CascadeConfig& cfg)
    : cascade_(cfg)
{
}

void BaselineExtractor::Collect(const RamanSplit& split)
{
    if (split.spectra.size() != split.count * kN
        || split.baselines.size() != split.count * kN)
    {
        throw std::invalid_argument(
            "BaselineExtractor::Collect: count does not match buffer size");
    }

    RamanNorm::Check();

    std::vector<float> spectra(split.spectra.size());
    std::vector<float> baselines(split.baselines.size());
    std::vector<float> back(kN);
    for (size_t i = 0; i < split.count; ++i)
    {
        const auto spec = split.Spectrum(i);
        const auto lab = split.Baseline(i);
        const auto nrm = RamanNorm::FromSpectrum(spec);
        auto mapped = std::span<float>{spectra.data() + i * kN, kN};
        nrm.Apply(spec, mapped);
        nrm.Apply(lab, {baselines.data() + i * kN, kN});
        nrm.Invert(mapped, back);
        for (size_t j = 0; j < kN; ++j)
        {
            if (std::fabs(back[j] - spec[j]) > 1e-2f)
            {
                throw std::runtime_error(
                    "BaselineExtractor::Collect: spectrum denorm round-trip failed");
            }
        }
    }
    cascade_.CollectBatch(spectra, baselines);
}

void BaselineExtractor::Train()
{
    cascade_.TrainOnCollected();
}

void BaselineExtractor::Predict(std::span<const float> spectrum,
                                std::span<float> baseline)
{
    if (spectrum.size() != cascade_.N()
        || baseline.size() != cascade_.NumOutputs())
    {
        throw std::invalid_argument(
            "BaselineExtractor::Predict: spectrum and baseline size must equal N");
    }

    const auto nrm = RamanNorm::FromSpectrum(spectrum);
    std::vector<float> xn(cascade_.N());
    nrm.Apply(spectrum, xn);
    const auto yn = cascade_.Predict(xn);
    nrm.Invert(yn, baseline);
}

void BaselineExtractor::SaveReadout(const std::string& path_stem) const
{
    if (path_stem.empty())
    {
        throw std::invalid_argument(
            "BaselineExtractor::SaveReadout: empty path_stem");
    }
    if (!cascade_.readout().IsTrained())
    {
        throw std::logic_error(
            "BaselineExtractor::SaveReadout: readout is not trained");
    }

    namespace fs = std::filesystem;
    const fs::path stem(path_stem);
    if (stem.has_parent_path())
        fs::create_directories(stem.parent_path());

    // Write beside the destination, then replace. A crash mid-write must not
    // truncate a previous good checkpoint. Windows rename will not overwrite.
    const std::string tmp = stem.string() + ".writing";
    cascade_.readout().SaveHcnnModel(tmp);

    const fs::path src_hcnw = tmp + ".hcnw";
    const fs::path src_arch = tmp + ".arch.json";
    const fs::path dst_hcnw = stem.string() + ".hcnw";
    const fs::path dst_arch = stem.string() + ".arch.json";

    auto replace_file = [](const fs::path& src, const fs::path& dst) {
        std::error_code ec;
        fs::remove(dst, ec);
        fs::rename(src, dst);
    };
    replace_file(src_hcnw, dst_hcnw);
    replace_file(src_arch, dst_arch);
}

void BaselineExtractor::LoadReadout(const std::string& path_stem)
{
    if (path_stem.empty())
    {
        throw std::invalid_argument(
            "BaselineExtractor::LoadReadout: empty path_stem");
    }
    cascade_.readout().LoadHcnnModel(path_stem, ReadoutLoadMode::Eval);
}

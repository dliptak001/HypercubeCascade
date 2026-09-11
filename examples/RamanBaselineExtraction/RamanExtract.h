#pragma once

#include "BaselineExtractor.h"
#include "RamanDataset.h"

#include <filesystem>
#include <span>
#include <string>

void LoadExtractor(BaselineExtractor& ex, const std::string& stem);

void ExtractSplit(BaselineExtractor& ex, const RamanSplit& split,
                  std::span<float> baselines_out);

void WriteRamanRow(const std::filesystem::path& path,
                   std::span<const float> row);

void WritePredictions(const std::filesystem::path& out_dir,
                      std::span<const int> indices,
                      std::span<const float> baselines_flat,
                      const std::string& stem,
                      const std::filesystem::path& split_dir);

#pragma once

// MNIST data location for demos. Fixed deploy root only — never walks the
// CLion / source-tree clone. Example-only helper.
//
//   C:\HypercubeCascade\data

#include <filesystem>
#include <stdexcept>

namespace cascade_ex {

inline bool DirHasMnistIdx(const std::filesystem::path& dir)
{
    namespace fs = std::filesystem;
    return fs::is_directory(dir)
           && fs::exists(dir / "train-images-idx3-ubyte")
           && fs::exists(dir / "train-labels-idx1-ubyte")
           && fs::exists(dir / "t10k-images-idx3-ubyte")
           && fs::exists(dir / "t10k-labels-idx1-ubyte");
}

/// Resolve C:\HypercubeCascade\data when it contains the four MNIST IDX files.
/// @p argv0 unused (kept for call-site stability).
inline std::filesystem::path FindMnistDataDir(const char* /*argv0*/)
{
    namespace fs = std::filesystem;
    const fs::path data("C:/HypercubeCascade/data");
    if (DirHasMnistIdx(data))
        return fs::weakly_canonical(data);

    throw std::runtime_error(
        "Cannot find MNIST IDX files.\n"
        "Looked in:\n"
        "  C:\\HypercubeCascade\\data\n"
        "Need:\n"
        "  train-images-idx3-ubyte  train-labels-idx1-ubyte\n"
        "  t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte");
}

} // namespace cascade_ex

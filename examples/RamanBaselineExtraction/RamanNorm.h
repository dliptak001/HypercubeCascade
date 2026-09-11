#pragma once

#include <algorithm>
#include <span>
#include <stdexcept>

/// Per-spectrum min/max of the **input**, mapped to [-1, 1]:
///   u    = (x - min) / range
///   norm = 2 * u - 1
///   x    = (norm + 1) * 0.5 * range + min
/// Labels use the matching .data min/range, never their own.
struct RamanNorm
{
    float min = 0.0f;
    float range = 1.0f;

    static RamanNorm FromSpectrum(std::span<const float> spectrum)
    {
        if (spectrum.empty())
            throw std::invalid_argument("RamanNorm: empty spectrum");

        float lo = spectrum[0];
        float hi = spectrum[0];
        for (const float v : spectrum)
        {
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }

        RamanNorm n;
        n.min = lo;
        n.range = hi - lo;
        return n;
    }

    void Apply(std::span<const float> in, std::span<float> out) const
    {
        if (in.size() != out.size())
            throw std::invalid_argument("RamanNorm::Apply: size mismatch");
        if (range == 0.0f)
        {
            for (size_t i = 0; i < in.size(); ++i)
                out[i] = 0.0f;
            return;
        }
        const float s = 2.0f / range;
        for (size_t i = 0; i < in.size(); ++i)
            out[i] = (in[i] - min) * s - 1.0f;
    }

    void Invert(std::span<const float> in, std::span<float> out) const
    {
        if (in.size() != out.size())
            throw std::invalid_argument("RamanNorm::Invert: size mismatch");
        const float s = 0.5f * range;
        for (size_t i = 0; i < in.size(); ++i)
            out[i] = (in[i] + 1.0f) * s + min;
    }

    static void Check()
    {
        const float data[] = {100.0f, 200.0f, 400.0f, 500.0f};
        const auto n = FromSpectrum(data);
        if (n.min != 100.0f || n.range != 400.0f)
            throw std::logic_error("RamanNorm::Check: min/range");

        float mapped[4] = {};
        float back[4] = {};
        n.Apply(data, mapped);
        if (mapped[0] != -1.0f || mapped[3] != 1.0f
            || mapped[1] != -0.5f || mapped[2] != 0.5f)
        {
            throw std::logic_error("RamanNorm::Check: [-1, 1] map");
        }
        n.Invert(mapped, back);
        for (int i = 0; i < 4; ++i)
        {
            if (back[i] != data[i])
                throw std::logic_error("RamanNorm::Check: invert");
        }

        const float flat[] = {7.0f, 7.0f, 7.0f};
        const auto z = FromSpectrum(flat);
        float zmap[3] = {};
        float zback[3] = {};
        z.Apply(flat, zmap);
        z.Invert(zmap, zback);
        float junk[3] = {0.9f, -0.3f, 1.0f};
        float junk_back[3] = {};
        z.Invert(junk, junk_back);
        for (int i = 0; i < 3; ++i)
        {
            if (z.range != 0.0f || zmap[i] != 0.0f || zback[i] != 7.0f
                || junk_back[i] != 7.0f)
            {
                throw std::logic_error("RamanNorm::Check: flat");
            }
        }
    }
};

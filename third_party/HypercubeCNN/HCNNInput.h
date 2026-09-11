// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

/**
 * @file HCNNInput.h
 * @brief Full-capacity packed inputs for HCNN train / infer (smell-6 fix).
 *
 * **Contract:** every sample is exactly `capacity` floats, where
 * `capacity = input_channels * N` and `N = 2^start_dim` for the network.
 *
 * Why this exists: spatial embed may pad unused vertices with a non-zero
 * `pad_value` (e.g. −1).  Network `Embed` always **zero-pads** when
 * `input_length < capacity`, which would wipe that padding.  These types
 * force the safe path: you only ever pass **full-capacity** buffers into
 * the typed overloads.
 *
 * Raw `float* + input_length` APIs remain for power users and intentional
 * short + zero-pad packing via `HCNNInputBatch::from_short_zero_pad`.
 */

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hcnn {

// =============================================================================
// Non-owning full-capacity view
// =============================================================================

/**
 * Non-owning view of `count` samples × `capacity` floats (row-major).
 * Does not allocate.  Lifetime is the caller's buffer.
 */
class HCNNInputView {
public:
    HCNNInputView() = default;

    /**
     * View over an already-full buffer.
     * @param flat      Base pointer; length at least count * capacity
     * @param count     Number of samples (>= 0)
     * @param capacity  Per-sample length; must match network capacity
     * @throws std::invalid_argument if flat is null when count*capacity > 0,
     *         or count/capacity negative, or capacity == 0 when count > 0
     */
    static HCNNInputView from_full(const float* flat, int count, int capacity) {
        HCNNInputView v;
        v.set_full_(flat, count, capacity);
        return v;
    }

    [[nodiscard]] const float* data() const { return data_; }
    [[nodiscard]] int count() const { return count_; }
    /// Per-sample length; always the full capacity (never a short pattern P).
    [[nodiscard]] int capacity() const { return capacity_; }
    /// Alias for capacity() — use as TrainEpoch input_length.
    [[nodiscard]] int input_length() const { return capacity_; }
    [[nodiscard]] bool empty() const { return count_ == 0; }

    /// Pointer to sample `i` (length `capacity()`).  No bounds check.
    [[nodiscard]] const float* sample(int i) const {
        return data_ + static_cast<size_t>(i) * static_cast<size_t>(capacity_);
    }

    /// Throws if capacity != expected (e.g. net channels * GetStartN()).
    void require_capacity(int expected) const {
        if (capacity_ != expected) {
            throw std::invalid_argument(
                "HCNNInputView: capacity " + std::to_string(capacity_)
                + " != expected " + std::to_string(expected));
        }
    }

private:
    const float* data_ = nullptr;
    int count_ = 0;
    int capacity_ = 0;

    void set_full_(const float* flat, int count, int capacity) {
        if (count < 0 || capacity < 0) {
            throw std::invalid_argument(
                "HCNNInputView: count and capacity must be >= 0");
        }
        if (count > 0 && capacity == 0) {
            throw std::invalid_argument(
                "HCNNInputView: capacity must be > 0 when count > 0");
        }
        const size_t n =
            static_cast<size_t>(count) * static_cast<size_t>(capacity);
        if (n > 0 && flat == nullptr) {
            throw std::invalid_argument("HCNNInputView: flat is null");
        }
        data_ = flat;
        count_ = count;
        capacity_ = capacity;
    }
};

// =============================================================================
// Owning full-capacity batch
// =============================================================================

/**
 * Owning buffer of `count` full-capacity samples.
 * Convert to a view with `view()` for HCNN typed overloads.
 */
class HCNNInputBatch {
public:
    HCNNInputBatch() = default;

    /// Allocate `count * capacity` floats (contents undefined).
    void reset(int count, int capacity) {
        if (count < 0 || capacity < 0) {
            throw std::invalid_argument(
                "HCNNInputBatch::reset: count and capacity must be >= 0");
        }
        if (count > 0 && capacity == 0) {
            throw std::invalid_argument(
                "HCNNInputBatch::reset: capacity must be > 0 when count > 0");
        }
        std::vector<float> neu(
            static_cast<size_t>(count) * static_cast<size_t>(capacity));
        data_.swap(neu);
        count_ = count;
        capacity_ = capacity;
    }

    /// Copy from an already-full row-major buffer (size must be count*capacity).
    static HCNNInputBatch from_full(const float* flat, int count, int capacity) {
        if (count < 0 || capacity < 0) {
            throw std::invalid_argument(
                "HCNNInputBatch::from_full: count and capacity must be >= 0");
        }
        const size_t n =
            static_cast<size_t>(count) * static_cast<size_t>(capacity);
        if (n > 0 && flat == nullptr) {
            throw std::invalid_argument("HCNNInputBatch::from_full: flat is null");
        }
        HCNNInputBatch b;
        b.reset(count, capacity);
        if (n > 0)
            std::memcpy(b.data_.data(), flat, n * sizeof(float));
        return b;
    }

    /**
     * Take ownership of a vector.  Throws if
     * `flat.size() != count * capacity`.
     */
    static HCNNInputBatch adopt(std::vector<float> flat, int count, int capacity) {
        if (count < 0 || capacity < 0) {
            throw std::invalid_argument(
                "HCNNInputBatch::adopt: count and capacity must be >= 0");
        }
        const size_t need =
            static_cast<size_t>(count) * static_cast<size_t>(capacity);
        if (flat.size() != need) {
            throw std::invalid_argument(
                "HCNNInputBatch::adopt: size " + std::to_string(flat.size())
                + " != count*capacity " + std::to_string(need));
        }
        HCNNInputBatch b;
        b.data_ = std::move(flat);
        b.count_ = count;
        b.capacity_ = capacity;
        return b;
    }

    /**
     * Explicit short → full path: per sample copy `input_length` floats and
     * **zero-pad** the tail to `capacity`.  Use when you want network-style
     * zero fill (native cube packing), not spatial `pad_value`.
     *
     * `flat` is row-major with stride `input_length` (not capacity).
     */
    static HCNNInputBatch from_short_zero_pad(const float* flat, int count,
                                              int input_length, int capacity) {
        if (count < 0 || input_length < 0 || capacity < 0) {
            throw std::invalid_argument(
                "HCNNInputBatch::from_short_zero_pad: negative size");
        }
        if (input_length > capacity) {
            throw std::invalid_argument(
                "HCNNInputBatch::from_short_zero_pad: input_length > capacity");
        }
        if (count > 0 && (flat == nullptr || capacity == 0)) {
            throw std::invalid_argument(
                "HCNNInputBatch::from_short_zero_pad: null flat or zero capacity");
        }
        HCNNInputBatch b;
        b.reset(count, capacity);
        for (int i = 0; i < count; ++i) {
            float* dst = b.sample(i);
            const float* src =
                flat + static_cast<size_t>(i) * static_cast<size_t>(input_length);
            if (input_length > 0)
                std::memcpy(dst, src, static_cast<size_t>(input_length) * sizeof(float));
            if (input_length < capacity) {
                std::memset(dst + input_length, 0,
                            static_cast<size_t>(capacity - input_length) * sizeof(float));
            }
        }
        return b;
    }

    [[nodiscard]] float* data() { return data_.data(); }
    [[nodiscard]] const float* data() const { return data_.data(); }
    [[nodiscard]] int count() const { return count_; }
    [[nodiscard]] int capacity() const { return capacity_; }
    [[nodiscard]] int input_length() const { return capacity_; }
    [[nodiscard]] bool empty() const { return count_ == 0; }
    [[nodiscard]] size_t size() const { return data_.size(); }

    [[nodiscard]] float* sample(int i) {
        return data_.data()
            + static_cast<size_t>(i) * static_cast<size_t>(capacity_);
    }
    [[nodiscard]] const float* sample(int i) const {
        return data_.data()
            + static_cast<size_t>(i) * static_cast<size_t>(capacity_);
    }

    [[nodiscard]] HCNNInputView view() const {
        return HCNNInputView::from_full(data_.data(), count_, capacity_);
    }

    void require_capacity(int expected) const {
        view().require_capacity(expected);
    }

private:
    std::vector<float> data_;
    int count_ = 0;
    int capacity_ = 0;
};

} // namespace hcnn

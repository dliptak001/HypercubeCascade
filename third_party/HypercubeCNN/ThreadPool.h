// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace hcnn {

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)  // structure was padded due to alignment specifier
#endif

/**
 * @class ThreadPool
 * @brief Minimal fork-join thread pool for `parallel_for`-style workloads.
 *
 * Workers are spawned once at construction and reused across `ForEach`
 * calls; there is no task queue, no work stealing, no priorities.  The
 * calling thread participates as thread 0, so a pool of N workers gives
 * N+1 active threads during a `ForEach`.
 *
 * `ForEach(count, func)` partitions `[0, count)` into N+1 contiguous
 * chunks and invokes `func(thread_id, range_begin, range_end)` once per
 * chunk.  Blocks until every chunk completes.  Exceptions thrown from any
 * chunk are captured and rethrown on the calling thread after the join.
 *
 * Constraints:
 *   - **Not reentrant.**  Calling `ForEach` from inside another `ForEach`
 *     on the same pool deadlocks.  Higher-level code (HCNNNetwork) uses
 *     `LayerThreadGuard` to disable per-layer vertex threading during
 *     batch dispatch precisely to prevent this.
 *   - Must be driven from a single thread; concurrent `ForEach` calls on
 *     the same pool are unsupported.
 *   - Do not destroy the pool while a `ForEach` is in progress.
 *   - Non-copyable.  Move is not provided (the worker threads capture
 *     `this`).
 *
 * Exceptions: worker exceptions are captured and rethrown on the caller
 * after all workers finish.  If the caller's own chunk throws, workers are
 * still joined before the exception propagates (no abandoned threads /
 * dangling functor references).
 *
 * Header-only, no dependencies beyond `<thread>` / `<mutex>` / `<atomic>`.
 *
 * Private implementation — not installed.  Used by HCNNNetwork and in-tree
 * tests only.  Application code must not include this header.
 */
class ThreadPool
{
public:
    /// Create pool with num_workers background threads (0 = auto-detect).
    explicit ThreadPool(size_t num_workers = 0)
    {
        if (num_workers == 0)
        {
            const unsigned hw = std::thread::hardware_concurrency();
            num_workers = (hw > 1) ? hw - 1 : 0;
        }

        workers_.reserve(num_workers);
        for (size_t i = 0; i < num_workers; ++i)
            workers_.emplace_back([this, i] { WorkerLoop(i + 1); });
    }

    ~ThreadPool()
    {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto& w : workers_)
            w.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Total thread count (workers + caller).
    [[nodiscard]] size_t NumThreads() const { return workers_.size() + 1; }

    /// Number of background worker threads (excludes caller).
    [[nodiscard]] size_t NumWorkers() const { return workers_.size(); }

    /// Execute func(thread_id, range_begin, range_end) for chunks of [0, count).
    /// Caller participates as thread 0. Blocks until all work completes.
    /// Must be called from a single thread (not concurrently with itself).
    template <typename F>
    void ForEach(size_t count, F&& func)
    {
        if (count == 0) return;

        if (workers_.empty())
        {
            func(size_t{0}, size_t{0}, count);
            return;
        }

        const size_t nt = NumThreads();
        const size_t chunk = (count + nt - 1) / nt;

        // Publish work for background workers
        {
            std::lock_guard lock(mutex_);
            exception_ = nullptr;
            for_func_ = [&func, chunk, count](size_t tid) {
                const size_t b = tid * chunk;
                if (b >= count) return;
                func(tid, b, std::min(b + chunk, count));
            };
            remaining_.store(static_cast<int>(workers_.size()));
            ++generation_;
        }
        cv_work_.notify_all();

        // Caller handles chunk 0.  Capture any throw so we still join workers
        // (for_func_ holds a reference to `func` until remaining_ hits 0).
        std::exception_ptr caller_ex;
        try {
            func(size_t{0}, size_t{0}, std::min(chunk, count));
        } catch (...) {
            caller_ex = std::current_exception();
        }

        // Always wait for all workers before releasing for_func_ / returning.
        {
            std::unique_lock lock(mutex_);
            cv_done_.wait(lock, [this] { return remaining_.load() == 0; });
            for_func_ = nullptr; // release captured references
            if (caller_ex) {
                exception_ = nullptr;
                std::rethrow_exception(caller_ex);
            }
            if (exception_) std::rethrow_exception(exception_);
        }
    }

private:
    void WorkerLoop(size_t tid)
    {
        size_t local_gen = 0;
        std::function<void(size_t)> fn;  // reused across iterations (avoids per-call heap alloc)
        while (true)
        {
            {
                std::unique_lock lock(mutex_);
                cv_work_.wait(lock, [&] { return stop_ || generation_ > local_gen; });
                if (stop_) return;
                local_gen = generation_;
                fn = for_func_;   // copy under lock to avoid race with ForEach clearing it
            }

            try {
                fn(tid);
            } catch (...) {
                std::lock_guard elock(mutex_);
                if (!exception_)
                    exception_ = std::current_exception();
            }

            if (remaining_.fetch_sub(1) == 1) {
                // Acquire the mutex once before notifying to close the
                // lost-wakeup race against ForEach's cv_done_.wait(lock, pred).
                // Without this, a caller that reads a stale `remaining_` under
                // its mutex hold, then commits to wait(), can be added to the
                // CV waiter list AFTER we've already notified — and notify
                // with an empty waiter list is a no-op.  Taking the mutex here
                // blocks until the caller has either already seen remaining_==0
                // or fully entered wait(), making the subsequent notify race-free.
                { std::lock_guard lock(mutex_); }
                cv_done_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;

    std::function<void(size_t)> for_func_;
    std::exception_ptr exception_;
    size_t generation_ = 0;
    bool stop_ = false;

    // remaining_ on its own cache line to avoid false sharing with mutex/cv.
    alignas(64) std::atomic<int> remaining_{0};
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace hcnn

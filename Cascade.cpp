#include "Cascade.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace {

uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

} // namespace

// ---------------------------------------------------------------------------
// Persistent collect thread pool
//
// Background workers live for the Cascade lifetime. Each ForEach is fork-join:
// the calling thread is tid 0; workers 1..nthreads-1 take the other chunks.
// Extra parked workers (pool larger than this job) wait out the generation
// without touching active_.
//
// std::thread + mutex/cv only. No OpenMP.
// Not re-entrant: do not call ForEach from a callback that already runs
// inside ForEach on this pool.
// ---------------------------------------------------------------------------

struct Cascade::CollectPool
{
    explicit CollectPool(size_t background_workers)
    {
        workers_.reserve(background_workers);
        for (size_t i = 0; i < background_workers; ++i)
            workers_.emplace_back([this, i] { WorkerLoop(i + 1); });
    }

    ~CollectPool()
    {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto& w : workers_)
            w.join();
    }

    CollectPool(const CollectPool&) = delete;
    CollectPool& operator=(const CollectPool&) = delete;

    [[nodiscard]] size_t NumThreads() const { return workers_.size() + 1; }

    /// @p func(tid, begin, end) over [0, count). Blocks until all done.
    /// No heap: the callable stays on the caller's stack; workers see a
    /// function pointer + context pointer.
    template <typename F>
    void ForEach(size_t count, size_t nthreads, F&& func)
    {
        if (count == 0)
            return;

        nthreads = std::max<size_t>(1, std::min({nthreads, count, NumThreads()}));
        if (nthreads == 1)
        {
            func(size_t{0}, size_t{0}, count);
            return;
        }

        const size_t chunk = (count + nthreads - 1) / nthreads;
        const int bg = static_cast<int>(nthreads - 1);

        struct Adapt
        {
            std::remove_reference_t<F>* f;
            size_t chunk;
            size_t count;
            size_t nthreads;
        } adapt{&func, chunk, count, nthreads};

        {
            std::lock_guard lock(mutex_);
            exception_ = nullptr;
            job_nthreads_ = nthreads;
            active_.store(bg);
            job_ctx_ = &adapt;
            job_fn_ = [](void* ctx, size_t tid) {
                auto* a = static_cast<Adapt*>(ctx);
                if (tid >= a->nthreads)
                    return;
                const size_t b = tid * a->chunk;
                if (b >= a->count)
                    return;
                (*a->f)(tid, b, std::min(b + a->chunk, a->count));
            };
            ++generation_;
        }
        cv_work_.notify_all();

        std::exception_ptr caller_ex;
        try
        {
            func(size_t{0}, size_t{0}, std::min(chunk, count));
        }
        catch (...)
        {
            caller_ex = std::current_exception();
        }

        {
            std::unique_lock lock(mutex_);
            cv_done_.wait(lock, [this] { return active_.load() == 0; });
            job_fn_ = nullptr;
            job_ctx_ = nullptr;
            if (caller_ex)
            {
                exception_ = nullptr;
                std::rethrow_exception(caller_ex);
            }
            if (exception_)
                std::rethrow_exception(exception_);
        }
    }

private:
    using JobFn = void (*)(void* ctx, size_t tid);

    void WorkerLoop(size_t tid)
    {
        size_t local_gen = 0;
        JobFn fn = nullptr;
        void* ctx = nullptr;
        size_t job_nt = 0;
        while (true)
        {
            {
                std::unique_lock lock(mutex_);
                cv_work_.wait(lock, [&] { return stop_ || generation_ > local_gen; });
                if (stop_)
                    return;
                local_gen = generation_;
                fn = job_fn_;
                ctx = job_ctx_;
                job_nt = job_nthreads_;
            }

            if (tid < job_nt && fn)
            {
                try
                {
                    fn(ctx, tid);
                }
                catch (...)
                {
                    std::lock_guard elock(mutex_);
                    if (!exception_)
                        exception_ = std::current_exception();
                }

                if (active_.fetch_sub(1) == 1)
                {
                    { std::lock_guard lock(mutex_); }
                    cv_done_.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;

    JobFn job_fn_ = nullptr;
    void* job_ctx_ = nullptr;
    std::exception_ptr exception_;
    size_t generation_ = 0;
    size_t job_nthreads_ = 0;
    bool stop_ = false;
    alignas(64) std::atomic<int> active_{0};
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

Cascade::Cascade(const CascadeConfig& cfg)
    : cfg_(cfg)
{
    if (cfg_.dim < 5 || cfg_.dim > 12)
    {
        throw std::invalid_argument(
            "Cascade: dim must be in 5 <= dim <= 12");
    }
    if (cfg_.T < 1)
        throw std::invalid_argument("Cascade: T must be >= 1");
    if (!(cfg_.interstage_scale > 0.0f) || !std::isfinite(cfg_.interstage_scale))
    {
        throw std::invalid_argument(
            "Cascade: interstage_scale must be finite and > 0");
    }
    if (!(cfg_.readout_scale > 0.0f) || !std::isfinite(cfg_.readout_scale))
    {
        throw std::invalid_argument(
            "Cascade: readout_scale must be finite and > 0");
    }
    if (cfg_.readout.num_outputs < 1)
        throw std::invalid_argument("Cascade: readout.num_outputs must be >= 1");

    cfg_.exciter.dim = cfg_.dim;
    cfg_.reservoir.dim = cfg_.dim;
    cfg_.readout.dim = cfg_.dim;
    if (cfg_.exciter.subcube_dim < 1 || cfg_.exciter.subcube_dim > cfg_.dim)
    {
        throw std::invalid_argument(
            "Cascade: exciter.subcube_dim must be in 1 <= subcube_dim <= dim");
    }

    exciter_ = Exciter::Create(cfg_.exciter);
    reservoir_ = Reservoir::Create(cfg_.reservoir);
    dim_ = exciter_->Dim();
    n_ = exciter_->N();
    if (reservoir_->Dim() != dim_ || reservoir_->Size() != n_)
    {
        throw std::logic_error(
            "Cascade: Reservoir size does not match Exciter");
    }

    readout_ = std::make_unique<Readout>(cfg_.readout);
    if (readout_->NumFeatures() != n_)
    {
        throw std::logic_error(
            "Cascade: readout NumFeatures does not match N = 2^dim");
    }

    const size_t M = reservoir_->HistoryDepth();
    s0_.assign(n_ * M, 0.0f);
    std::mt19937_64 rng(mix64(cfg_.ic_seed ^ 0x5343000000000001ULL));
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (float& v : s0_)
        v = dist(rng);

    last_exciter_.reserve(n_);
    last_interstage_.reserve(n_);
    last_reservoir_.reserve(n_);
    last_features_.reserve(n_);
    ClearCollected();

    CollectWorker primary;
    primary.ex = exciter_.get();
    primary.res = reservoir_.get();
    primary.field.assign(n_, 0.0f);
    primary.transit.assign(n_, 0.0f);
    primary.excited.assign(n_, 0.0f);
    primary.drive.assign(n_, 0.0f);
    collect_workers_.push_back(std::move(primary));
}

Cascade::~Cascade() = default;

// ---------------------------------------------------------------------------
// Collect workers / pool
// ---------------------------------------------------------------------------

size_t Cascade::ResolveCollectThreads(size_t count) const
{
    if (count == 0)
        return 1;
    size_t n = cfg_.collect_threads;
    if (n == 0)
    {
        const unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0)
        {
            n = 1;
        }
        else
        {
            const unsigned reserve = (hw >= 8u) ? 2u : 1u;
            n = (hw > reserve) ? static_cast<size_t>(hw - reserve) : 1u;
        }
    }
    if (n < 1)
        n = 1;
    return std::min(n, count);
}

void Cascade::EnsureCollectWorkers(size_t n)
{
    if (n <= collect_workers_.size())
        return;

    const size_t already = collect_workers_.size();
    const size_t need = n - already;
    std::vector<CollectWorker> fresh(need);
    std::exception_ptr ex;
    std::mutex ex_mu;

    auto build_one = [&](size_t k) {
        try
        {
            ReservoirConfig rc = cfg_.reservoir;
            rc.verbose = false;
            CollectWorker w;
            w.owned_ex = Exciter::Create(cfg_.exciter);
            w.ex = w.owned_ex.get();
            w.owned_res = Reservoir::Create(rc);
            w.res = w.owned_res.get();
            w.field.assign(n_, 0.0f);
            w.transit.assign(n_, 0.0f);
            w.excited.assign(n_, 0.0f);
            w.drive.assign(n_, 0.0f);
            fresh[k] = std::move(w);
        }
        catch (...)
        {
            std::lock_guard lock(ex_mu);
            if (!ex)
                ex = std::current_exception();
        }
    };

    if (need == 1)
    {
        build_one(0);
    }
    else
    {
        std::vector<std::thread> thr;
        thr.reserve(need);
        for (size_t k = 0; k < need; ++k)
            thr.emplace_back([&, k] { build_one(k); });
        for (auto& t : thr)
            t.join();
    }
    if (ex)
        std::rethrow_exception(ex);

    collect_workers_.reserve(n);
    for (auto& w : fresh)
        collect_workers_.push_back(std::move(w));
}

void Cascade::EnsureCollectPool(size_t nthreads)
{
    if (nthreads <= 1)
        return;
    const size_t want_bg = nthreads - 1;
    if (!collect_pool_ || collect_pool_->NumThreads() < nthreads)
        collect_pool_ = std::make_unique<CollectPool>(want_bg);
}

void Cascade::PublishLastRow(const float* rows, size_t count)
{
    if (count == 0 || rows == nullptr)
        return;
    const float* last = rows + (count - 1) * n_;
    last_features_.assign(last, last + n_);
}

void Cascade::MapOn(CollectWorker& w, const float* x, float* dest) const
{
    if (w.ex == nullptr || w.res == nullptr)
        throw std::logic_error("Cascade::MapOn: null stage");

    std::memcpy(w.field.data(), x, n_ * sizeof(float));
    const float* y = w.ex->ExciteCube(w.field.data());
    std::memcpy(w.transit.data(), y, n_ * sizeof(float));
    const float scale = cfg_.interstage_scale;
    for (size_t i = 0; i < n_; ++i)
        w.excited[i] = y[i] * scale;

    w.res->LoadInitialCondition(s0_.data(), s0_.size());

    const size_t n_mask = n_ - 1;
    size_t c = 0;
    for (size_t pass = 0; pass < cfg_.T; ++pass)
    {
        for (size_t v = 0; v < n_; ++v)
            w.drive[v] = w.excited[(v ^ c) & n_mask];
        w.res->InjectInputField(w.drive.data(), n_);
        w.res->Step();
        ++c;
    }

    const float* z = w.res->Outputs();
    const float out_scale = cfg_.readout_scale;
    for (size_t i = 0; i < n_; ++i)
        dest[i] = z[i] * out_scale;
}

void Cascade::MapFeaturesParallel(const float* fields_flat, size_t count,
                                  float* dest)
{
    if (count == 0)
        return;
    if (fields_flat == nullptr || dest == nullptr)
        throw std::logic_error("Cascade::MapFeaturesParallel: null buffer");

    const size_t nw = ResolveCollectThreads(count);
    EnsureCollectWorkers(nw);
    EnsureCollectPool(nw);

    auto run_range = [&](size_t tid, size_t begin, size_t end) {
        CollectWorker& w = collect_workers_[tid];
        for (size_t i = begin; i < end; ++i)
            MapOn(w, fields_flat + i * n_, dest + i * n_);
    };

    if (nw <= 1 || !collect_pool_)
        run_range(0, 0, count);
    else
        collect_pool_->ForEach(count, nw, run_range);
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

void Cascade::MapInto(std::span<const float> x, float* dest)
{
    if (x.size() != n_)
    {
        throw std::invalid_argument(
            "Cascade: field size must equal N = 2^dim");
    }
    CollectWorker& w = collect_workers_[0];
    MapOn(w, x.data(), dest);
    last_exciter_.assign(w.transit.begin(), w.transit.end());
    last_interstage_.assign(w.excited.begin(), w.excited.end());
    last_reservoir_.assign(w.res->Outputs(), w.res->Outputs() + n_);
}

void Cascade::Run(std::span<const float> x)
{
    last_features_.resize(n_);
    MapInto(x, last_features_.data());
}

void Cascade::MapBatchInto(std::span<const float> fields_flat,
                           std::vector<float>& out_features)
{
    if (fields_flat.size() % n_ != 0)
    {
        throw std::invalid_argument(
            "Cascade: fields_flat length must be a multiple of N");
    }

    const size_t count = fields_flat.size() / n_;
    out_features.resize(count * n_);
    if (count == 0)
        return;

    MapFeaturesParallel(fields_flat.data(), count, out_features.data());
    PublishLastRow(out_features.data(), count);
}

// ---------------------------------------------------------------------------
// Collect
// ---------------------------------------------------------------------------

void Cascade::ClearCollected()
{
    collected_features_.clear();
    collected_labels_.clear();
    collected_targets_.clear();
    num_collected_ = 0;
}

void Cascade::RequireClassification() const
{
    if (cfg_.readout.task != ReadoutTask::Classification)
    {
        throw std::invalid_argument(
            "Cascade: classification API used but task is Regression");
    }
}

void Cascade::RequireRegression() const
{
    if (cfg_.readout.task != ReadoutTask::Regression)
    {
        throw std::invalid_argument(
            "Cascade: regression API used but task is Classification");
    }
}

void Cascade::MapCollectedOne(std::span<const float> x)
{
    last_features_.resize(n_);
    MapInto(x, last_features_.data());
}

void Cascade::Collect(std::span<const float> x, int class_label)
{
    RequireClassification();
    if (class_label < 0 || class_label >= cfg_.readout.num_outputs)
    {
        throw std::invalid_argument(
            "Cascade::Collect: class_label must be in [0, num_outputs)");
    }

    MapCollectedOne(x);

    const size_t old_feat = collected_features_.size();
    try
    {
        collected_features_.insert(collected_features_.end(),
                                   last_features_.begin(), last_features_.end());
        collected_labels_.push_back(class_label);
    }
    catch (...)
    {
        collected_features_.resize(old_feat);
        collected_labels_.resize(num_collected_);
        throw;
    }
    ++num_collected_;
}

void Cascade::Collect(std::span<const float> x, std::span<const float> target)
{
    RequireRegression();
    if (target.size() != static_cast<size_t>(cfg_.readout.num_outputs))
    {
        throw std::invalid_argument(
            "Cascade::Collect: target size must equal num_outputs");
    }

    MapCollectedOne(x);

    const size_t old_feat = collected_features_.size();
    const size_t old_tgt = collected_targets_.size();
    try
    {
        collected_features_.insert(collected_features_.end(),
                                   last_features_.begin(), last_features_.end());
        collected_targets_.insert(collected_targets_.end(), target.begin(),
                                  target.end());
    }
    catch (...)
    {
        collected_features_.resize(old_feat);
        collected_targets_.resize(old_tgt);
        throw;
    }
    ++num_collected_;
}

void Cascade::CollectBatch(std::span<const float> fields_flat,
                           std::span<const int> labels)
{
    RequireClassification();
    if (fields_flat.size() % n_ != 0)
    {
        throw std::invalid_argument(
            "Cascade::CollectBatch: fields_flat length must be a multiple of N");
    }
    const size_t count = fields_flat.size() / n_;
    if (labels.size() != count)
    {
        throw std::invalid_argument(
            "Cascade::CollectBatch: labels.size() must equal field count");
    }
    for (size_t i = 0; i < count; ++i)
    {
        if (labels[i] < 0 || labels[i] >= cfg_.readout.num_outputs)
        {
            throw std::invalid_argument(
                "Cascade::CollectBatch: label out of range");
        }
    }
    if (count == 0)
        return;

    const size_t base = num_collected_;
    const size_t old_feat = collected_features_.size();
    const size_t old_lab = collected_labels_.size();
    try
    {
        collected_labels_.resize(base + count);
        std::memcpy(collected_labels_.data() + base, labels.data(),
                    count * sizeof(int));
        collected_features_.resize((base + count) * n_);
        MapFeaturesParallel(fields_flat.data(), count,
                            collected_features_.data() + base * n_);
    }
    catch (...)
    {
        collected_features_.resize(old_feat);
        collected_labels_.resize(old_lab);
        throw;
    }
    num_collected_ = base + count;
    PublishLastRow(collected_features_.data() + base * n_, count);
}

void Cascade::CollectBatch(std::span<const float> fields_flat,
                           std::span<const float> targets_flat)
{
    RequireRegression();
    if (fields_flat.size() % n_ != 0)
    {
        throw std::invalid_argument(
            "Cascade::CollectBatch: fields_flat length must be a multiple of N");
    }
    const size_t count = fields_flat.size() / n_;
    const size_t no = static_cast<size_t>(cfg_.readout.num_outputs);
    if (targets_flat.size() != count * no)
    {
        throw std::invalid_argument(
            "Cascade::CollectBatch: targets_flat length must equal "
            "count * num_outputs");
    }
    if (count == 0)
        return;

    const size_t base = num_collected_;
    const size_t old_feat = collected_features_.size();
    const size_t old_tgt = collected_targets_.size();
    try
    {
        collected_targets_.resize((base + count) * no);
        std::memcpy(collected_targets_.data() + base * no, targets_flat.data(),
                    count * no * sizeof(float));
        collected_features_.resize((base + count) * n_);
        MapFeaturesParallel(fields_flat.data(), count,
                            collected_features_.data() + base * n_);
    }
    catch (...)
    {
        collected_features_.resize(old_feat);
        collected_targets_.resize(old_tgt);
        throw;
    }
    num_collected_ = base + count;
    PublishLastRow(collected_features_.data() + base * n_, count);
}

void Cascade::TrainOnCollected()
{
    if (num_collected_ == 0)
    {
        throw std::invalid_argument(
            "Cascade::TrainOnCollected: no samples collected");
    }

    if (cfg_.readout.task == ReadoutTask::Classification)
    {
        if (collected_labels_.size() != num_collected_)
        {
            throw std::logic_error(
                "Cascade::TrainOnCollected: label buffer size mismatch");
        }
        readout_->Train(collected_features_.data(), collected_labels_.data(),
                        num_collected_);
    }
    else
    {
        const size_t no = static_cast<size_t>(cfg_.readout.num_outputs);
        if (collected_targets_.size() != num_collected_ * no)
        {
            throw std::logic_error(
                "Cascade::TrainOnCollected: target buffer size mismatch");
        }
        readout_->Train(collected_features_.data(), collected_targets_.data(),
                        num_collected_);
    }
}

// ---------------------------------------------------------------------------
// Inference / metrics
// ---------------------------------------------------------------------------

std::vector<float> Cascade::Predict(std::span<const float> x)
{
    Run(x);
    std::vector<float> out(NumOutputs());
    readout_->PredictRaw(last_features_.data(), out.data());
    return out;
}

int Cascade::PredictClass(std::span<const float> x)
{
    RequireClassification();
    Run(x);
    return readout_->PredictClass(last_features_.data());
}

double Cascade::AccuracyOnCollected() const
{
    RequireClassification();
    if (num_collected_ == 0)
    {
        throw std::invalid_argument(
            "Cascade::AccuracyOnCollected: no samples");
    }
    return readout_->Accuracy(collected_features_.data(),
                              collected_labels_.data(), num_collected_);
}

double Cascade::R2OnCollected() const
{
    RequireRegression();
    if (num_collected_ == 0)
    {
        throw std::invalid_argument(
            "Cascade::R2OnCollected: no samples");
    }
    return readout_->R2(collected_features_.data(), collected_targets_.data(),
                        num_collected_);
}

double Cascade::Accuracy(std::span<const float> fields_flat,
                         std::span<const int> labels)
{
    RequireClassification();
    if (fields_flat.size() % n_ != 0)
    {
        throw std::invalid_argument(
            "Cascade::Accuracy: fields_flat length must be a multiple of N");
    }
    const size_t count = fields_flat.size() / n_;
    if (labels.size() != count)
    {
        throw std::invalid_argument(
            "Cascade::Accuracy: labels.size() must equal field count");
    }
    if (count == 0)
    {
        throw std::invalid_argument(
            "Cascade::Accuracy: no samples");
    }

    for (size_t i = 0; i < count; ++i)
    {
        if (labels[i] < 0 || labels[i] >= cfg_.readout.num_outputs)
        {
            throw std::invalid_argument(
                "Cascade::Accuracy: label out of range");
        }
    }

    MapBatchInto(fields_flat, eval_features_);
    return readout_->Accuracy(eval_features_.data(), labels.data(), count);
}

double Cascade::R2(std::span<const float> fields_flat,
                   std::span<const float> targets_flat)
{
    RequireRegression();
    if (fields_flat.size() % n_ != 0)
    {
        throw std::invalid_argument(
            "Cascade::R2: fields_flat length must be a multiple of N");
    }
    const size_t count = fields_flat.size() / n_;
    const size_t no = static_cast<size_t>(cfg_.readout.num_outputs);
    if (targets_flat.size() != count * no)
    {
        throw std::invalid_argument(
            "Cascade::R2: targets_flat length must equal count * num_outputs");
    }
    if (count == 0)
    {
        throw std::invalid_argument(
            "Cascade::R2: no samples");
    }

    MapBatchInto(fields_flat, eval_features_);
    return readout_->R2(eval_features_.data(), targets_flat.data(), count);
}

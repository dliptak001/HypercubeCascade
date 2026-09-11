// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak
//
// Thin pybind11 surface for HypercubeCascade. Ergonomics (shape checks, fit,
// pickle, docs) live in hypercube_cascade/__init__.py.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Cascade.h"

namespace py = pybind11;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

namespace {

void require_field_size(size_t got, size_t n, const char* what)
{
    if (got != n)
        throw std::invalid_argument(
            std::string(what) + " size (" + std::to_string(got)
            + ") must equal N (" + std::to_string(n) + ")");
}

py::array_t<float> span_to_array(std::span<const float> span)
{
    py::array_t<float> arr(span.size());
    if (!span.empty())
        std::memcpy(arr.mutable_data(), span.data(), span.size() * sizeof(float));
    return arr;
}

} // namespace

// Single de-templated Cascade binding. Hypercube dim is a runtime constructor
// argument (cfg.dim, stamped onto all three stages), so one C++ type and one
// Python class serve every dimension 5–12 — no per-DIM instantiations.
PYBIND11_MODULE(_core, m)
{
    m.doc() = "HypercubeCascade: frozen etalon transit + frozen reservoir orbit "
              "+ HypercubeCNN on end state";
#ifndef HYPERCUBE_CASCADE_VERSION
#  error "HYPERCUBE_CASCADE_VERSION must be set by CMake from hypercube_cascade/_version.py"
#endif
    m.attr("__version__") = HYPERCUBE_CASCADE_VERSION;

    py::class_<Cascade>(m, "_Cascade")
        // ── Construction ──
        // All cascade + exciter + reservoir + readout parameters fixed at
        // construction. cfg.dim is stamped onto every stage.
        .def(py::init([](size_t dim, size_t T, float interstage_scale,
                         float readout_scale, uint64_t ic_seed,
                         size_t collect_threads,
                         uint64_t exciter_seed, float exciter_input_scaling,
                         float exciter_weight_scaling, size_t exciter_subcube_dim,
                         uint64_t reservoir_seed, float spectral_radius,
                         float reservoir_input_scaling, float leak_rate,
                         size_t history_depth, float bias_scaling, bool verbose,
                         int readout_num_outputs, const char* readout_task,
                         int readout_num_layers, int readout_conv_channels,
                         int readout_epochs, int readout_batch_size,
                         float readout_lr_max, float readout_lr_min_frac,
                         int readout_lr_decay_epochs, float readout_weight_decay,
                         float readout_momentum, const char* readout_activation,
                         uint64_t readout_seed, size_t readout_num_threads,
                         bool readout_restore_best_epoch,
                         float readout_best_epoch_holdout_frac,
                         bool readout_use_pooling) {
            CascadeConfig cfg;
            cfg.dim              = dim;
            cfg.T                = T;
            cfg.interstage_scale = interstage_scale;
            cfg.readout_scale    = readout_scale;
            cfg.ic_seed          = ic_seed;
            cfg.collect_threads  = collect_threads;
            cfg.exciter.seed           = exciter_seed;
            cfg.exciter.input_scaling  = exciter_input_scaling;
            cfg.exciter.weight_scaling = exciter_weight_scaling;
            cfg.exciter.subcube_dim    = exciter_subcube_dim;
            cfg.reservoir.seed            = reservoir_seed;
            cfg.reservoir.spectral_radius = spectral_radius;
            cfg.reservoir.input_scaling   = reservoir_input_scaling;
            cfg.reservoir.leak_rate       = leak_rate;
            cfg.reservoir.history_depth   = history_depth;
            cfg.reservoir.bias_scaling    = bias_scaling;
            cfg.reservoir.verbose         = verbose;
            cfg.readout.num_outputs = readout_num_outputs;
            cfg.readout.task = (std::strcmp(readout_task, "classification") == 0)
                                   ? ReadoutTask::Classification
                                   : ReadoutTask::Regression;
            if (std::strcmp(readout_task, "classification") != 0
                && std::strcmp(readout_task, "regression") != 0) {
                throw std::invalid_argument(
                    std::string("readout_task must be 'classification' or "
                                "'regression' (got '")
                    + readout_task + "')");
            }
            cfg.readout.num_layers      = readout_num_layers;
            cfg.readout.conv_channels   = readout_conv_channels;
            cfg.readout.epochs          = readout_epochs;
            cfg.readout.batch_size      = readout_batch_size;
            cfg.readout.lr_max          = readout_lr_max;
            cfg.readout.lr_min_frac     = readout_lr_min_frac;
            cfg.readout.lr_decay_epochs = readout_lr_decay_epochs;
            cfg.readout.weight_decay    = readout_weight_decay;
            cfg.readout.momentum        = readout_momentum;
            if      (std::strcmp(readout_activation, "relu") == 0)
                cfg.readout.activation = ReadoutActivation::RELU;
            else if (std::strcmp(readout_activation, "leaky_relu") == 0)
                cfg.readout.activation = ReadoutActivation::LEAKY_RELU;
            else if (std::strcmp(readout_activation, "none") == 0)
                cfg.readout.activation = ReadoutActivation::NONE;
            else if (std::strcmp(readout_activation, "tanh") == 0)
                cfg.readout.activation = ReadoutActivation::TANH;
            else
                throw std::invalid_argument(
                    std::string("readout_activation must be one of "
                                "'tanh', 'relu', 'leaky_relu', 'none' (got '")
                    + readout_activation + "')");
            cfg.readout.seed                    = readout_seed;
            cfg.readout.num_threads             = readout_num_threads;
            cfg.readout.restore_best_epoch      = readout_restore_best_epoch;
            cfg.readout.best_epoch_holdout_frac = readout_best_epoch_holdout_frac;
            cfg.readout.use_pooling             = readout_use_pooling;
            return std::make_unique<Cascade>(cfg);
        }),
            py::arg("dim"),
            py::arg("T")                        = 100ULL,
            py::arg("interstage_scale")         = 1.0f,
            py::arg("readout_scale")            = 1.0f,
            py::arg("ic_seed")                  = 1ULL,
            py::arg("collect_threads")          = 0ULL,
            py::arg("exciter_seed")             = 7934791766227647176ULL,
            py::arg("exciter_input_scaling")    = 0.02f,
            py::arg("exciter_weight_scaling")   = 0.02f,
            py::arg("exciter_subcube_dim")      = 6ULL,
            py::arg("reservoir_seed")           = 7934791766227647176ULL,
            py::arg("spectral_radius")          = 0.999f,
            py::arg("reservoir_input_scaling")  = 0.02f,
            py::arg("leak_rate")                = 1.0f,
            py::arg("history_depth")            = 16ULL,
            py::arg("bias_scaling")             = 0.003f,
            py::arg("verbose")                  = false,
            py::arg("readout_num_outputs")      = 1,
            py::arg("readout_task")             = "regression",
            py::arg("readout_num_layers")       = 1,
            py::arg("readout_conv_channels")    = 16,
            py::arg("readout_epochs")           = 200,
            py::arg("readout_batch_size")       = 32,
            py::arg("readout_lr_max")           = 0.0015f,
            py::arg("readout_lr_min_frac")      = 0.01f,
            py::arg("readout_lr_decay_epochs")  = 0,
            py::arg("readout_weight_decay")     = 0.0f,
            py::arg("readout_momentum")         = 0.9f,
            py::arg("readout_activation")       = "tanh",
            py::arg("readout_seed")             = 42ULL,
            py::arg("readout_num_threads")      = 0ULL,
            py::arg("readout_restore_best_epoch") = true,
            py::arg("readout_best_epoch_holdout_frac") = 0.0f,
            py::arg("readout_use_pooling")      = true)

        // ── Map (no training) ──
        .def("run", [](Cascade& self, FloatArray x) {
            auto buf = x.request();
            require_field_size(static_cast<size_t>(buf.size), self.N(), "field");
            py::gil_scoped_release release;
            self.Run({static_cast<const float*>(buf.ptr), self.N()});
        }, py::arg("x"),
           "Map one field: transit -> orbit -> features. Updates last_* probes.")

        .def("last_features", [](const Cascade& self) {
            return span_to_array(self.LastFeatures());
        }, "Features (length N) from the most recent completed map.")

        .def("last_exciter", [](const Cascade& self) {
            return span_to_array(self.LastExciter());
        }, "Etalon transit output from the last serial map. Not updated by bulk calls.")

        .def("last_interstage", [](const Cascade& self) {
            return span_to_array(self.LastInterstage());
        }, "Transit output * interstage_scale (the reservoir drive) from the "
           "last serial map. Not updated by bulk calls.")

        .def("last_reservoir", [](const Cascade& self) {
            return span_to_array(self.LastReservoir());
        }, "Reservoir end state (before readout_scale) from the last serial map. "
           "Not updated by bulk calls.")

        .def("clear_collected", &Cascade::ClearCollected,
             "Drop all samples collected for batch training.")

        // ── Collect (serial) ──
        .def("collect_class", [](Cascade& self, FloatArray x, int class_label) {
            auto buf = x.request();
            require_field_size(static_cast<size_t>(buf.size), self.N(), "field");
            py::gil_scoped_release release;
            self.Collect(
                {static_cast<const float*>(buf.ptr), self.N()}, class_label);
        }, py::arg("x"), py::arg("class_label"),
           "Serial collect one classification sample.")

        .def("collect_reg", [](Cascade& self, FloatArray x, FloatArray target) {
            auto xbuf = x.request();
            auto tbuf = target.request();
            require_field_size(static_cast<size_t>(xbuf.size), self.N(), "field");
            if (static_cast<size_t>(tbuf.size) != self.NumOutputs())
                throw std::invalid_argument(
                    "target size (" + std::to_string(tbuf.size)
                    + ") must equal num_outputs (" + std::to_string(self.NumOutputs())
                    + ")");
            py::gil_scoped_release release;
            self.Collect(
                {static_cast<const float*>(xbuf.ptr), self.N()},
                std::span<const float>{static_cast<const float*>(tbuf.ptr),
                                       static_cast<size_t>(tbuf.size)});
        }, py::arg("x"), py::arg("target"),
           "Serial collect one regression sample.")

        // ── Collect (bulk) ──
        .def("collect_batch_class", [](Cascade& self, FloatArray fields, IntArray labels) {
            auto fbuf = fields.request();
            auto lbuf = labels.request();
            const size_t n = self.N();
            const size_t total = static_cast<size_t>(fbuf.size);
            if (total % n != 0)
                throw std::invalid_argument(
                    "fields size (" + std::to_string(total)
                    + ") must be a multiple of N (" + std::to_string(n) + ")");
            const size_t count = total / n;
            if (static_cast<size_t>(lbuf.size) != count)
                throw std::invalid_argument(
                    "labels length (" + std::to_string(lbuf.size)
                    + ") must equal sample count (" + std::to_string(count) + ")");
            py::gil_scoped_release release;
            self.CollectBatch(
                {static_cast<const float*>(fbuf.ptr), total},
                {static_cast<const int*>(lbuf.ptr), count});
        }, py::arg("fields"), py::arg("labels"),
           "Bulk parallel collect (classification). fields: (count, N) or flat count*N.")

        .def("collect_batch_reg", [](Cascade& self, FloatArray fields, FloatArray targets) {
            auto fbuf = fields.request();
            auto tbuf = targets.request();
            const size_t n = self.N();
            const size_t k = self.NumOutputs();
            const size_t total = static_cast<size_t>(fbuf.size);
            if (total % n != 0)
                throw std::invalid_argument(
                    "fields size (" + std::to_string(total)
                    + ") must be a multiple of N (" + std::to_string(n) + ")");
            const size_t count = total / n;
            if (static_cast<size_t>(tbuf.size) != count * k)
                throw std::invalid_argument(
                    "targets size (" + std::to_string(tbuf.size)
                    + ") must equal count * num_outputs ("
                    + std::to_string(count * k) + ")");
            py::gil_scoped_release release;
            self.CollectBatch(
                {static_cast<const float*>(fbuf.ptr), total},
                std::span<const float>{static_cast<const float*>(tbuf.ptr),
                                       static_cast<size_t>(tbuf.size)});
        }, py::arg("fields"), py::arg("targets"),
           "Bulk parallel collect (regression). fields: (count, N); "
           "targets: (count, num_outputs) or flat.")

        // ── Train / predict ──
        .def("train", [](Cascade& self) {
            py::gil_scoped_release release;
            self.TrainOnCollected();
        }, "Batch-train the HCNN on all collected samples.")

        .def("predict", [](Cascade& self, FloatArray x) {
            auto buf = x.request();
            require_field_size(static_cast<size_t>(buf.size), self.N(), "field");
            std::vector<float> out;
            {
                py::gil_scoped_release release;
                out = self.Predict({static_cast<const float*>(buf.ptr), self.N()});
            }
            py::array_t<float> arr(out.size());
            std::memcpy(arr.mutable_data(), out.data(), out.size() * sizeof(float));
            return arr;
        }, py::arg("x"),
           "Fresh map + readout forward; returns (num_outputs,) float32.")

        .def("predict_class", [](Cascade& self, FloatArray x) {
            auto buf = x.request();
            require_field_size(static_cast<size_t>(buf.size), self.N(), "field");
            py::gil_scoped_release release;
            return self.PredictClass({static_cast<const float*>(buf.ptr), self.N()});
        }, py::arg("x"),
           "Fresh map + argmax class (classification only).")

        .def("accuracy_on_collected", [](const Cascade& self) {
            py::gil_scoped_release release;
            return self.AccuracyOnCollected();
        }, "Train-set accuracy on collected samples (not a test metric).")

        .def("r2_on_collected", [](const Cascade& self) {
            py::gil_scoped_release release;
            return self.R2OnCollected();
        }, "Train-set R^2 on collected samples (not a test metric).")

        // ── Held-out scoring (fresh maps) ──
        .def("accuracy", [](Cascade& self, FloatArray fields, IntArray labels) {
            auto fbuf = fields.request();
            auto lbuf = labels.request();
            const size_t n = self.N();
            const size_t total = static_cast<size_t>(fbuf.size);
            if (total % n != 0)
                throw std::invalid_argument(
                    "fields size (" + std::to_string(total)
                    + ") must be a multiple of N (" + std::to_string(n) + ")");
            const size_t count = total / n;
            if (static_cast<size_t>(lbuf.size) != count)
                throw std::invalid_argument(
                    "labels length (" + std::to_string(lbuf.size)
                    + ") must equal sample count (" + std::to_string(count) + ")");
            py::gil_scoped_release release;
            return self.Accuracy(
                {static_cast<const float*>(fbuf.ptr), total},
                {static_cast<const int*>(lbuf.ptr), count});
        }, py::arg("fields"), py::arg("labels"),
           "Fresh map + accuracy on a caller-owned set (bulk, parallel).")

        .def("r2", [](Cascade& self, FloatArray fields, FloatArray targets) {
            auto fbuf = fields.request();
            auto tbuf = targets.request();
            const size_t n = self.N();
            const size_t k = self.NumOutputs();
            const size_t total = static_cast<size_t>(fbuf.size);
            if (total % n != 0)
                throw std::invalid_argument(
                    "fields size (" + std::to_string(total)
                    + ") must be a multiple of N (" + std::to_string(n) + ")");
            const size_t count = total / n;
            if (static_cast<size_t>(tbuf.size) != count * k)
                throw std::invalid_argument(
                    "targets size (" + std::to_string(tbuf.size)
                    + ") must equal count * num_outputs ("
                    + std::to_string(count * k) + ")");
            py::gil_scoped_release release;
            return self.R2(
                {static_cast<const float*>(fbuf.ptr), total},
                std::span<const float>{static_cast<const float*>(tbuf.ptr),
                                       static_cast<size_t>(tbuf.size)});
        }, py::arg("fields"), py::arg("targets"),
           "Fresh map + R^2 on a caller-owned set (bulk, parallel).")

        // ── Properties ──
        .def_property_readonly("dim", &Cascade::Dim)
        .def_property_readonly("N", &Cascade::N)
        .def_property_readonly("num_collected", &Cascade::NumCollected)
        .def_property_readonly("num_outputs", &Cascade::NumOutputs)
        .def_property_readonly("collect_threads", &Cascade::CollectThreads)
        .def_property_readonly("T", [](const Cascade& self) {
            return self.config().T;
        })
        .def_property_readonly("interstage_scale", [](const Cascade& self) {
            return self.config().interstage_scale;
        })
        .def_property_readonly("readout_scale", [](const Cascade& self) {
            return self.config().readout_scale;
        })
        .def_property_readonly("ic_seed", [](const Cascade& self) {
            return self.config().ic_seed;
        })
        .def_property_readonly("exciter_seed", [](const Cascade& self) {
            return self.exciter().GetConfig().seed;
        })
        .def_property_readonly("exciter_input_scaling", [](const Cascade& self) {
            return self.exciter().GetConfig().input_scaling;
        })
        .def_property_readonly("exciter_weight_scaling", [](const Cascade& self) {
            return self.exciter().GetConfig().weight_scaling;
        })
        .def_property_readonly("subcube_dim", [](const Cascade& self) {
            return self.exciter().SubcubeDim();
        })
        .def_property_readonly("walk_size", [](const Cascade& self) {
            return self.exciter().WalkSize();
        })
        .def_property_readonly("reservoir_seed", [](const Cascade& self) {
            return self.reservoir().GetConfig().seed;
        })
        .def_property_readonly("spectral_radius", [](const Cascade& self) {
            return self.reservoir().GetConfig().spectral_radius;
        })
        .def_property_readonly("realized_spectral_radius", [](const Cascade& self) {
            return self.reservoir().GetRealizedSpectralRadius();
        })
        .def_property_readonly("reservoir_input_scaling", [](const Cascade& self) {
            return self.reservoir().GetConfig().input_scaling;
        })
        .def_property_readonly("leak_rate", [](const Cascade& self) {
            return self.reservoir().GetConfig().leak_rate;
        })
        .def_property_readonly("history_depth", [](const Cascade& self) {
            return self.reservoir().GetConfig().history_depth;
        })
        .def_property_readonly("bias_scaling", [](const Cascade& self) {
            return self.reservoir().GetConfig().bias_scaling;
        })
        .def_property_readonly("readout_task", [](const Cascade& self) {
            return self.readout().GetConfig().task == ReadoutTask::Classification
                       ? "classification"
                       : "regression";
        })
        .def_property_readonly("readout_best_epoch", [](const Cascade& self) {
            return self.readout().BestEpoch();
        })

        // ── Persistence helpers ──
        .def("_get_readout_state", [](const Cascade& self) -> py::dict {
            py::dict d;
            d["is_trained"] = self.readout().IsTrained();
            auto w = self.readout().Weights();
            d["weights"] = py::array_t<double>(
                {static_cast<py::ssize_t>(w.size())}, w.data());
            return d;
        })
        .def("_set_readout_state", [](Cascade& self, py::dict d) {
            if (!d.contains("is_trained") || !d["is_trained"].cast<bool>())
                return;
            auto w = d["weights"].cast<
                py::array_t<double, py::array::c_style | py::array::forcecast>>();
            std::vector<double> weights(w.data(), w.data() + w.size());
            ReadoutLoadMode mode = ReadoutLoadMode::Eval;
            if (d.contains("mode")) {
                const auto ms = d["mode"].cast<std::string>();
                if (ms == "resume_train" || ms == "ResumeTrain")
                    mode = ReadoutLoadMode::ResumeTrain;
            }
            self.readout().SetState(std::move(weights), mode);
        })
        .def("save_readout_hcnn_model",
             [](const Cascade& self, const std::string& path_stem) {
                 self.readout().SaveHcnnModel(path_stem);
             },
             py::arg("path_stem"),
             "Write portable stem.hcnw + stem.arch.json for the HCNN readout.")
        .def("load_readout_hcnn_model",
             [](Cascade& self, const std::string& path_stem, const std::string& mode) {
                 ReadoutLoadMode m = ReadoutLoadMode::Eval;
                 if (mode == "resume_train" || mode == "ResumeTrain")
                     m = ReadoutLoadMode::ResumeTrain;
                 self.readout().LoadHcnnModel(path_stem, m);
             },
             py::arg("path_stem"),
             py::arg("mode") = "eval",
             "Load stem.hcnw (+ arch sidecar) into the live readout.")
        .def("readout_arch_summary",
             [](const Cascade& self) { return self.readout().ArchSummary(); },
             "Human-readable HCNN readout architecture and parameter counts.")
        ;
}

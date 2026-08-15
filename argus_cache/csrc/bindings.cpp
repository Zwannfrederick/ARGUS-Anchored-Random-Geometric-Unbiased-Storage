#include "manager.h"
#include "tier_codec.h"
#include "zero_copy_pool.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // ── Tier codec registry ────────────────────────────────────────────────
    // Exposing CodecKind/TierCodec is what lets a Python plugin claim a real
    // native storage format for its tier instead of being limited to the
    // uncompressed passthrough fallback.
    py::enum_<argus::CodecKind>(m, "CodecKind")
        .value("SIGNED_LINEAR", argus::CodecKind::SignedLinear)
        .value("UNSIGNED_AFFINE", argus::CodecKind::UnsignedAffine)
        .value("SIGN_PACKED", argus::CodecKind::SignPacked)
        .value("PROJECTION", argus::CodecKind::Projection)
        .value("PASSTHROUGH", argus::CodecKind::Passthrough);

    py::class_<argus::TierCodec>(m, "TierCodec")
        .def(py::init<>())
        .def_readwrite("name", &argus::TierCodec::name)
        .def_readwrite("kind", &argus::TierCodec::kind)
        .def_readwrite("bits", &argus::TierCodec::bits)
        .def_readwrite("lossy", &argus::TierCodec::lossy)
        .def_readwrite("compression_ratio", &argus::TierCodec::compression_ratio)
        .def_property_readonly("pack_factor", &argus::TierCodec::pack_factor)
        .def_property_readonly("levels", &argus::TierCodec::levels)
        .def_property_readonly("needs_min", &argus::TierCodec::needs_min)
        .def_property_readonly("is_quantized", &argus::TierCodec::is_quantized)
        .def_property_readonly("effective_bits", &argus::TierCodec::effective_bits)
        .def("__repr__", [](const argus::TierCodec &c) {
            return "<TierCodec '" + c.name + "' bits=" + std::to_string(c.bits) +
                   " pack=" + std::to_string(c.pack_factor()) + ">";
        });

    m.def("create_page", []() { return std::make_shared<Page>(); }, "Create an empty Page");
    
    py::class_<Page, std::shared_ptr<Page>>(m, "Page")
        .def_readwrite("page_id", &Page::page_id)
        .def_readwrite("pool_slot", &Page::pool_slot)
        .def_readwrite("page_size", &Page::page_size)
        .def_readwrite("importance_score", &Page::importance_score)
        .def_readwrite("attention_sum", &Page::attention_sum)
        .def_readwrite("last_step_accessed", &Page::last_step_accessed)
        .def_readwrite("tier_name", &Page::tier_name)
        .def_readwrite("orig_dtype", &Page::orig_dtype)
        .def_readwrite("key", &Page::key_tensor)
        .def_readwrite("value", &Page::value_tensor)
        .def_readwrite("key_compressed", &Page::compressed_key)
        .def_readwrite("value_compressed", &Page::compressed_value)
        .def_readwrite("key_scale", &Page::key_scale)
        .def_readwrite("value_scale", &Page::value_scale)
        .def_readwrite("key_min", &Page::key_min)
        .def_readwrite("value_min", &Page::value_min)
        .def("__getitem__", [](Page& self, const std::string& key) -> py::object {
            if (key == "page_id") return py::cast(self.page_id);
            if (key == "pool_slot" || key == "pool_idx") return py::cast(self.pool_slot);
            if (key == "page_size") return py::cast(self.page_size);
            if (key == "importance_score") return py::cast(self.importance_score);
            if (key == "attention_sum") return py::cast(self.attention_sum);
            if (key == "last_step_accessed") return py::cast(self.last_step_accessed);
            if (key == "tier_name") return py::cast(self.tier_name);
            if (key == "orig_dtype" || key == "dtype") return py::cast(self.orig_dtype);
            if (key == "key") return self.key_tensor.defined() ? py::cast(self.key_tensor) : py::none();
            if (key == "value") return self.value_tensor.defined() ? py::cast(self.value_tensor) : py::none();
            if (key == "key_compressed") {
                if (!self.compressed_key.defined()) return py::none();
                py::dict d;
                d["q"] = self.compressed_key;
                d["scales"] = torch::tensor(self.key_scale);
                d["min_vals"] = torch::tensor(self.key_min);
                return d;
            }
            if (key == "value_compressed") {
                if (!self.compressed_value.defined()) return py::none();
                py::dict d;
                d["q"] = self.compressed_value;
                d["scales"] = torch::tensor(self.value_scale);
                d["min_vals"] = torch::tensor(self.value_min);
                return d;
            }
            if (key == "key_q" || key == "key_packed" || key == "key_proj")
                return self.compressed_key.defined() ? py::cast(self.compressed_key) : py::none();
            if (key == "value_q" || key == "value_packed" || key == "value_proj")
                return self.compressed_value.defined() ? py::cast(self.compressed_value) : py::none();
            if (key == "key_scale" || key == "key_scales") return py::cast(self.key_scale);
            if (key == "value_scale" || key == "value_scales") return py::cast(self.value_scale);
            if (key == "key_min") return py::cast(self.key_min);
            if (key == "value_min") return py::cast(self.value_min);
            auto it = self.extra_fields.find(key);
            if (it != self.extra_fields.end()) return it->second;
            throw py::key_error("key not found in Page");
        })
        .def("__setitem__", [](Page& self, const std::string& key, py::object value) {
            if (key == "page_id") self.page_id = py::cast<int>(value);
            else if (key == "pool_slot" || key == "pool_idx") self.pool_slot = py::cast<int>(value);
            else if (key == "page_size") self.page_size = py::cast<int>(value);
            else if (key == "importance_score") self.importance_score = py::cast<float>(value);
            else if (key == "attention_sum") self.attention_sum = py::cast<float>(value);
            else if (key == "last_step_accessed") self.last_step_accessed = py::cast<int>(value);
            else if (key == "tier_name") self.tier_name = py::cast<std::string>(value);
            else if (key == "orig_dtype" || key == "dtype") self.orig_dtype = py::cast<at::ScalarType>(value);
            else if (key == "key") self.key_tensor = py::cast<at::Tensor>(value);
            else if (key == "value") self.value_tensor = py::cast<at::Tensor>(value);
            else if (key == "key_compressed") self.compressed_key = py::cast<at::Tensor>(value);
            else if (key == "value_compressed") self.compressed_value = py::cast<at::Tensor>(value);
            else if (key == "key_scale" || key == "key_scales") self.key_scale = py::cast<float>(value);
            else if (key == "value_scale" || key == "value_scales") self.value_scale = py::cast<float>(value);
            else if (key == "key_min") self.key_min = py::cast<float>(value);
            else if (key == "value_min") self.value_min = py::cast<float>(value);
            else self.extra_fields[key] = value;
        })
        .def("get", [](Page& self, const std::string& key, py::object default_val) -> py::object {
            try {
                if (key == "page_id") return py::cast(self.page_id);
                if (key == "pool_slot" || key == "pool_idx") return py::cast(self.pool_slot);
                if (key == "page_size") return py::cast(self.page_size);
                if (key == "importance_score") return py::cast(self.importance_score);
                if (key == "attention_sum") return py::cast(self.attention_sum);
                if (key == "last_step_accessed") return py::cast(self.last_step_accessed);
                if (key == "tier_name") return py::cast(self.tier_name);
                if (key == "orig_dtype" || key == "dtype") return py::cast(self.orig_dtype);
                if (key == "key") return self.key_tensor.defined() ? py::cast(self.key_tensor) : default_val;
                if (key == "value") return self.value_tensor.defined() ? py::cast(self.value_tensor) : default_val;
                if (key == "key_compressed") {
                    if (!self.compressed_key.defined()) return default_val;
                    py::dict d;
                    d["q"] = self.compressed_key;
                    d["scales"] = torch::tensor(self.key_scale);
                    d["min_vals"] = torch::tensor(self.key_min);
                    return d;
                }
                if (key == "value_compressed") {
                    if (!self.compressed_value.defined()) return default_val;
                    py::dict d;
                    d["q"] = self.compressed_value;
                    d["scales"] = torch::tensor(self.value_scale);
                    d["min_vals"] = torch::tensor(self.value_min);
                    return d;
                }
                if (key == "key_q" || key == "key_packed" || key == "key_proj")
                    return self.compressed_key.defined() ? py::cast(self.compressed_key) : default_val;
                if (key == "value_q" || key == "value_packed" || key == "value_proj")
                    return self.compressed_value.defined() ? py::cast(self.compressed_value) : default_val;
                if (key == "key_scale" || key == "key_scales") return py::cast(self.key_scale);
                if (key == "value_scale" || key == "value_scales") return py::cast(self.value_scale);
                if (key == "key_min") return py::cast(self.key_min);
                if (key == "value_min") return py::cast(self.value_min);
                auto it = self.extra_fields.find(key);
                if (it != self.extra_fields.end()) return it->second;
            } catch (...) {}
            return default_val;
        }, py::arg("key"), py::arg("default_val") = py::none())
        .def("__contains__", [](Page& self, const std::string& key) -> bool {
            if (key == "page_id") return true;
            if (key == "pool_slot" || key == "pool_idx") return true;
            if (key == "page_size") return true;
            if (key == "importance_score") return true;
            if (key == "attention_sum") return true;
            if (key == "last_step_accessed") return true;
            if (key == "tier_name") return true;
            if (key == "orig_dtype" || key == "dtype") return true;
            if (key == "key") return self.key_tensor.defined();
            if (key == "value") return self.value_tensor.defined();
            if (key == "key_compressed") return self.compressed_key.defined();
            if (key == "value_compressed") return self.compressed_value.defined();
            if (key == "key_q" || key == "key_packed" || key == "key_proj") return self.compressed_key.defined();
            if (key == "value_q" || key == "value_packed" || key == "value_proj") return self.compressed_value.defined();
            if (key == "key_scale" || key == "key_scales") return true;
            if (key == "value_scale" || key == "value_scales") return true;
            if (key == "key_min") return true;
            if (key == "value_min") return true;
            return self.extra_fields.find(key) != self.extra_fields.end();
        });

    py::class_<ZeroCopyHostPool>(m, "ZeroCopyHostPool")
        .def(py::init<int>(), py::arg("device_id") = 0)
        .def("allocate", [](ZeroCopyHostPool& self, size_t size_bytes) -> py::object {
            void* ptr = self.allocate(size_bytes);
            if (!ptr) return py::none();
            return py::cast(reinterpret_cast<uintptr_t>(ptr));
        }, py::arg("size_bytes"))
        .def("free", [](ZeroCopyHostPool& self, py::object ptr_obj) {
            if (ptr_obj.is_none()) return;
            uintptr_t ptr_val = py::cast<uintptr_t>(ptr_obj);
            self.free(reinterpret_cast<void*>(ptr_val));
        }, py::arg("ptr"))
        .def("get_device_pointer", [](ZeroCopyHostPool& self, py::object host_ptr_obj) -> py::object {
            if (host_ptr_obj.is_none()) return py::none();
            uintptr_t host_ptr_val = py::cast<uintptr_t>(host_ptr_obj);
            void* dev_ptr = self.get_device_pointer(reinterpret_cast<void*>(host_ptr_val));
            if (!dev_ptr) return py::none();
            return py::cast(reinterpret_cast<uintptr_t>(dev_ptr));
        }, py::arg("host_ptr"))
        .def("tensor_to_pinned", [](ZeroCopyHostPool& self, const torch::Tensor& tensor) -> py::object {
            torch::Tensor pinned = self.tensor_to_pinned(tensor);
            py::object py_pinned = py::cast(pinned);
            size_t nbytes = tensor.numel() * tensor.element_size();
            void* raw_ptr = pinned.data_ptr();
            
            py_pinned.attr("_zero_copy_ptr") = py::cast(reinterpret_cast<uintptr_t>(raw_ptr));
            py_pinned.attr("_zero_copy_pool") = py::cast(&self, py::return_value_policy::reference);
            py_pinned.attr("_zero_copy_nbytes") = py::cast(nbytes);
            py_pinned.attr("_zero_copy_managed") = py::cast(self.is_zero_copy_tensor(pinned));
            
            return py_pinned;
        }, py::arg("tensor"))
        .def("free_tensor", &ZeroCopyHostPool::free_tensor, py::arg("tensor"))
        .def("is_zero_copy_tensor", &ZeroCopyHostPool::is_zero_copy_tensor, py::arg("tensor"))
        .def("get_fragmentation_report", &ZeroCopyHostPool::get_fragmentation_report)
        .def_readwrite("_total_allocated_bytes", &ZeroCopyHostPool::total_allocated_bytes_)
        .def_readwrite("_peak_allocated_bytes", &ZeroCopyHostPool::peak_allocated_bytes_)
        .def_readwrite("_fallback_mode", &ZeroCopyHostPool::fallback_mode_)
        .def_readwrite("_initialized", &ZeroCopyHostPool::initialized_)
        .def_readwrite("num_nodes", &ZeroCopyHostPool::num_nodes_)
        .def_readwrite("gpu_numa_node", &ZeroCopyHostPool::gpu_numa_node_);

    py::class_<ArgusCppManager>(m, "ArgusCppManager")
        .def(py::init<int, int, int>(), py::arg("page_size"), py::arg("max_active_pages"), py::arg("device_id"))
        .def("push_new_tokens", &ArgusCppManager::push_new_tokens, py::arg("k"), py::arg("v"))
        .def("inplace_paged_attention", &ArgusCppManager::inplace_paged_attention,
             py::arg("q"), py::arg("scale"),
             py::arg("sink_k") = c10::nullopt, py::arg("sink_v") = c10::nullopt,
             py::arg("anchor_k") = c10::nullopt, py::arg("anchor_v") = c10::nullopt,
             py::arg("k_buffer") = c10::nullopt, py::arg("v_buffer") = c10::nullopt,
             py::arg("resurrection_threshold") = 0.15f)
        .def("speculate_and_prefetch", &ArgusCppManager::speculate_and_prefetch, py::arg("page_ids"))
        .def("get_page_count", &ArgusCppManager::get_page_count)
        .def("get_active_page_count", &ArgusCppManager::get_active_page_count)
        .def("get_tier_page_count", &ArgusCppManager::get_tier_page_count, py::arg("tier_name"))
        .def("next_page_id", &ArgusCppManager::next_page_id)
        .def("get_prefetch_hit_count", &ArgusCppManager::get_prefetch_hit_count)
        .def("get_prefetched_page_ids", &ArgusCppManager::get_prefetched_page_ids)
        .def("get_prefetched_tensors", &ArgusCppManager::get_prefetched_tensors, py::arg("page_id"))
        .def("wait_for_prefetch_idle", &ArgusCppManager::wait_for_prefetch_idle)
        .def("get_host_pool", &ArgusCppManager::get_host_pool, py::return_value_policy::reference)
        .def("set_tier_max_pages", &ArgusCppManager::set_tier_max_pages, py::arg("tier_name"), py::arg("max_pages"))
        .def("resurrect_page", &ArgusCppManager::resurrect_page, py::arg("page"), py::arg("current_tier"))
        .def("peek_decompress_page", &ArgusCppManager::peek_decompress_page, py::arg("page"), py::arg("current_tier"))
        .def("manage_memory_lifecycle", &ArgusCppManager::manage_memory_lifecycle)
        .def("demote_to_next_tier", &ArgusCppManager::demote_to_next_tier, py::arg("page"))
        .def("add_tier_cpp", &ArgusCppManager::add_tier_cpp)
        .def("remove_tier_cpp", &ArgusCppManager::remove_tier_cpp)
        .def("set_tier_pipeline_cpp", &ArgusCppManager::set_tier_pipeline_cpp)
        .def("set_jl_projection_matrix", &ArgusCppManager::set_jl_projection_matrix, py::arg("w_proj"))
        .def("set_jl_recon_operator", &ArgusCppManager::set_jl_recon_operator, py::arg("recon_operator"))
        .def("set_active_pool_victim_selector", &ArgusCppManager::set_active_pool_victim_selector, py::arg("fn"))
        .def("set_on_page_access_callback", &ArgusCppManager::set_on_page_access_callback, py::arg("fn"))
        .def("set_jl_projection_provider", &ArgusCppManager::set_jl_projection_provider, py::arg("fn"))
        .def("set_jl_recon_provider", &ArgusCppManager::set_jl_recon_provider, py::arg("fn"))
        .def("set_force_qos", &ArgusCppManager::set_force_qos, py::arg("v"))
        .def("set_streaming_attention", &ArgusCppManager::set_streaming_attention,
             py::arg("v"))
        .def("set_verbose", &ArgusCppManager::set_verbose, py::arg("v"))
        .def("register_codec",
             [](ArgusCppManager &self, const argus::TierCodec &codec) {
                 self.codec_registry_.register_codec(codec);
             },
             py::arg("codec"),
             "Register or replace the native storage format for a tier name.")
        .def("unregister_codec",
             [](ArgusCppManager &self, const std::string &name) {
                 self.codec_registry_.unregister_codec(name);
             },
             py::arg("name"),
             "Drop a tier's native format; it reverts to uncompressed spill.")
        .def("has_codec",
             [](ArgusCppManager &self, const std::string &name) {
                 return self.codec_registry_.has(name);
             },
             py::arg("name"))
        .def("get_codec",
             [](ArgusCppManager &self, const std::string &name) {
                 return self.codec_registry_.get(name);
             },
             py::arg("name"))
        .def("codec_names",
             [](ArgusCppManager &self) { return self.codec_registry_.names(); })
        .def_readwrite("max_active_pages", &ArgusCppManager::max_active_pages_)
        .def_readwrite("generation_step", &ArgusCppManager::generation_step_)
        .def_property("active_pages", [](ArgusCppManager& self) -> std::vector<std::shared_ptr<Page>>& {
            return self.active_pages_;
        }, [](ArgusCppManager& self, const std::vector<std::shared_ptr<Page>>& pages) {
            self.active_pages_ = pages;
        }, py::return_value_policy::reference_internal)
        .def_property("pages_by_tier", [](ArgusCppManager& self) -> std::unordered_map<std::string, std::vector<std::shared_ptr<Page>>>& {
            return self.pages_by_tier_;
        }, [](ArgusCppManager& self, const std::unordered_map<std::string, std::vector<std::shared_ptr<Page>>>& pages) {
            self.pages_by_tier_ = pages;
        }, py::return_value_policy::reference_internal);
}

# ARGUS Experiments & Legacy Research Archive

Bu dizin, `libargus` (C++/CUDA GGML birleşik bellek ve bant genişliği orkestratörü) çekirdeğinin dışında kalan geçmiş araştırma yollarını ve uygulama deneylerini arşivler. Buradaki kodlar ana paketin (`argus_cache` / `libargus`) bir parçası değildir.

## İçerik

1. **`hermes/` ve `neo-mobile/`**: Yerel multimodal ajan (Hermes Supervisor, Cognitive Router) ve mobil istemci (Neo) entegrasyon deneyleri.
2. **`demo_train_and_generate.py`**: Erken dönem (v0.2–v0.3) sentetik eğitim ve üretim demosu.
3. **`Modelfile_*` ve `Dockerfile.*`**: Ollama ve konteyner tabanlı erken dönem karşılaştırma profilleri.
4. **HuggingFace Araştırma Yolu (`argus_cache/core/memory_manager.py` & `argus_cache/csrc/manager.cpp`)**:
   - v0.3–v0.4 döneminde geliştirilen Python/PyTorch tabanlı `PagedDynamicKVCache` ve çok katmanlı nicemleme (`fp8`, `int8`, `int4`, `int2`, `one_bit`, `jl`) araştırma motoru geriye dönük test uyumluluğu için `argus_cache` altında korunmaktadır.
   - Üretim ve yüksek performanslı orkestrasyon yolu **`libargus` (`argus_cache/csrc/ggml_*` ve `argus_config.h`)** C++/CUDA çekirdeğidir.


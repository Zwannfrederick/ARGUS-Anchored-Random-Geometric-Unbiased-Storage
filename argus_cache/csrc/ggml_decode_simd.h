// K1 internals: KV element types and the SIMD operations the decode kernels are written in.
// Not an API. Rows are addressed as (row pointer, element index); quantized rows are arrays
// of GGML blocks, laid out exactly as GGML stores them.
#pragma once

#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__AVX512F__) || (defined(__AVX2__) && defined(__FMA__) && defined(__F16C__))
#include <immintrin.h>
#define ARGUS_K1_SIMD 1
#endif

namespace argus_k1 {
struct BlockQ8_0 { ggml_fp16_t d; int8_t qs[32]; };
struct BlockQ4_0 { ggml_fp16_t d; uint8_t qs[16]; };
static_assert(sizeof(BlockQ8_0) == 34 && sizeof(BlockQ4_0) == 18, "GGML block layout");

template <typename T> constexpr int64_t elements_per_unit = 1;
template <> constexpr int64_t elements_per_unit<BlockQ8_0> = 32;
template <> constexpr int64_t elements_per_unit<BlockQ4_0> = 32;

template <typename T>
size_t row_bytes(int64_t d) { return d / elements_per_unit<T> * sizeof(T); }

inline float element(const float * row, int64_t i) { return row[i]; }
inline float element(const ggml_fp16_t * row, int64_t i) { return ggml_fp16_to_fp32(row[i]); }
inline float element(const BlockQ8_0 * row, int64_t i) {
    return ggml_fp16_to_fp32(row[i / 32].d) * row[i / 32].qs[i % 32];
}
inline float element(const BlockQ4_0 * row, int64_t i) {
    const BlockQ4_0 & b = row[i / 32];
    const int j = i % 32, q = j < 16 ? (b.qs[j] & 0x0F) : (b.qs[j - 16] >> 4);
    return ggml_fp16_to_fp32(b.d) * (q - 8);
}

#ifdef ARGUS_K1_SIMD
inline float half_to_float(ggml_fp16_t x) { return _cvtsh_ss(x); }

// Eight partial vectors to their eight lane sums, in order.
inline __m256 sums8(const __m256 * y) {
    const __m256 a = _mm256_hadd_ps(_mm256_hadd_ps(y[0], y[1]), _mm256_hadd_ps(y[2], y[3]));
    const __m256 b = _mm256_hadd_ps(_mm256_hadd_ps(y[4], y[5]), _mm256_hadd_ps(y[6], y[7]));
    return _mm256_add_ps(_mm256_permute2f128_ps(a, b, 0x20), _mm256_permute2f128_ps(a, b, 0x31));
}

#ifdef __AVX512F__
struct Simd {
    using V = __m512;
    static constexpr int W = 16;
    static V zero() { return _mm512_setzero_ps(); }
    static V load(const float * row, int64_t i) { return _mm512_loadu_ps(row + i); }
    static V load(const ggml_fp16_t * row, int64_t i) {
        return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + i)));
    }
    static V load(const BlockQ8_0 * row, int64_t i) {
        const BlockQ8_0 & b = row[i / 32];
        const __m512i q = _mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b.qs + i % 32)));
        return _mm512_mul_ps(_mm512_cvtepi32_ps(q), _mm512_set1_ps(half_to_float(b.d)));
    }
    static V load(const BlockQ4_0 * row, int64_t i) {
        const BlockQ4_0 & b = row[i / 32];
        __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b.qs));
        if (i % 32) { bytes = _mm_srli_epi16(bytes, 4); }
        const __m512i q = _mm512_cvtepu8_epi32(_mm_and_si128(bytes, _mm_set1_epi8(0x0F)));
        return _mm512_mul_ps(_mm512_sub_ps(_mm512_cvtepi32_ps(q), _mm512_set1_ps(8.0f)), _mm512_set1_ps(half_to_float(b.d)));
    }
    static void store(float * p, V x) { _mm512_storeu_ps(p, x); }
    static V set1(float x) { return _mm512_set1_ps(x); }
    static V fma(V a, V b, V c) { return _mm512_fmadd_ps(a, b, c); }
    static __m256 half(V x) { return _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1)); }
    // ggml_v_expf (ARM optimized routine): max error 1.45358 + 0.5 ulp; exp(-inf) = 0.
    static V exp(V x) {
        const V r = _mm512_set1_ps(0x1.8p23f);
        const V z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
        const V n = _mm512_sub_ps(z, r);
        const V b = _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f), _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
        const __mmask16 d = _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
        const V u = _mm512_mul_ps(b, b);
        const V j = _mm512_fmadd_ps(
            _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b, _mm512_set1_ps(0x1.573e2ep-5f)), u,
                            _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b, _mm512_set1_ps(0x1.fffdb6p-2f))),
            u, _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0f)));
        const V res = _mm512_scalef_ps(j, n);
        if (_mm512_kortestz(d, d)) { return res; }
        const V alt = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(n, _mm512_setzero_ps(), _CMP_LE_OQ),
                                           _mm512_set1_ps(INFINITY), _mm512_setzero_ps());
        return _mm512_mask_blend_ps(d, res, alt);
    }
    static float sum(V x) { return _mm512_reduce_add_ps(x); }
    static V max(V a, V b) { return _mm512_max_ps(a, b); }
    static float max(V x) { return _mm512_reduce_max_ps(x); }
};
#else
struct Simd {
    using V = __m256;
    static constexpr int W = 8;
    static V zero() { return _mm256_setzero_ps(); }
    static V load(const float * row, int64_t i) { return _mm256_loadu_ps(row + i); }
    static V load(const ggml_fp16_t * row, int64_t i) {
        return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(row + i)));
    }
    static V load(const BlockQ8_0 * row, int64_t i) {
        const BlockQ8_0 & b = row[i / 32];
        const __m256i q = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(b.qs + i % 32)));
        return _mm256_mul_ps(_mm256_cvtepi32_ps(q), _mm256_set1_ps(half_to_float(b.d)));
    }
    static V load(const BlockQ4_0 * row, int64_t i) {
        const BlockQ4_0 & b = row[i / 32];
        const int j = i % 32;
        __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(b.qs + j % 16));
        if (j >= 16) { bytes = _mm_srli_epi16(bytes, 4); }
        const __m256i q = _mm256_cvtepu8_epi32(_mm_and_si128(bytes, _mm_set1_epi8(0x0F)));
        return _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(q), _mm256_set1_ps(8.0f)), _mm256_set1_ps(half_to_float(b.d)));
    }
    static void store(float * p, V x) { _mm256_storeu_ps(p, x); }
    static V set1(float x) { return _mm256_set1_ps(x); }
    static V fma(V a, V b, V c) { return _mm256_fmadd_ps(a, b, c); }
    static __m256 half(V x) { return x; }
    // ggml_v_expf (ARM optimized routine): max error 1.45358 + 0.5 ulp; exp(-inf) = 0.
    static V exp(V x) {
        const V r = _mm256_set1_ps(0x1.8p23f);
        const V z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
        const V n = _mm256_sub_ps(z, r);
        const V b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f), _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
        const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
        const V k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
        const __m256i c = _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(126), _CMP_GT_OQ));
        const V u = _mm256_mul_ps(b, b);
        const V j = _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b, _mm256_set1_ps(0x1.573e2ep-5f)), u,
                            _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b, _mm256_set1_ps(0x1.fffdb6p-2f))),
            u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
        if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) { return _mm256_fmadd_ps(j, k, k); }
        const __m256i g = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
                                           _mm256_set1_epi32(0x82000000u));
        const V s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
        const V s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
        const __m256i d = _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(192), _CMP_GT_OQ));
        return _mm256_or_ps(
            _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
            _mm256_andnot_ps(_mm256_castsi256_ps(d),
                             _mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(c), _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                                          _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
    }
    static float sum(V x) {
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
    }
    static V max(V a, V b) { return _mm256_max_ps(a, b); }
    static float max(V x) {
        __m128 s = _mm_max_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
        s = _mm_max_ps(s, _mm_movehl_ps(s, s));
        return _mm_cvtss_f32(_mm_max_ss(s, _mm_movehdup_ps(s)));
    }
};
#endif
#endif
} // namespace argus_k1

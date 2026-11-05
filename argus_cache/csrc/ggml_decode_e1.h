// E1 + Multipole V-Centroid (NeurIPS 2025) + Top-r K-Bound Filtering (ShadowKV / SparQ)
// for CPU K1 decode attention.
#pragma once

#include "ggml_decode_attention.h"
#include "ggml_decode_simd.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace argus_k1 {

constexpr int64_t e1_block = 32;
constexpr float e1_minus_inf = -std::numeric_limits<float>::infinity();

// Per-query-row precomputed top-4 8-channel segments (32 channels) for Stage-1a fast filtering.
struct QueryTopR {
    int segs[4] = {0, 1, 2, 3};
    float q_tail_norm = 0.0f;
};

inline QueryTopR prepare_query_top_r(const float * q, int64_t d) {
    QueryTopR qr{};
    const int n_segs = static_cast<int>(d / 8);
    if (n_segs <= 4) { return qr; }
    float top_e[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    int top_i[4] = {0, 1, 2, 3};
    float total_sq = 0.0f;
    for (int s = 0; s < n_segs; ++s) {
        const float * qs = q + s * 8;
        float seg_sq = 0.0f;
        for (int j = 0; j < 8; ++j) { seg_sq += qs[j] * qs[j]; }
        total_sq += seg_sq;
        if (seg_sq > top_e[3]) {
            top_e[3] = seg_sq;
            top_i[3] = s;
            for (int k = 3; k > 0 && top_e[k] > top_e[k - 1]; --k) {
                std::swap(top_e[k], top_e[k - 1]);
                std::swap(top_i[k], top_i[k - 1]);
            }
        }
    }
    const float top_sq = std::max(0.0f, top_e[0]) + std::max(0.0f, top_e[1]) +
                         std::max(0.0f, top_e[2]) + std::max(0.0f, top_e[3]);
    for (int k = 0; k < 4; ++k) { qr.segs[k] = top_i[k]; }
    qr.q_tail_norm = std::sqrt(std::max(0.0f, total_sq - top_sq));
    return qr;
}

// Per-tensor lock-free bounded-mass bounding-box & multipole V-centroid sheet for 32-cell blocks.
struct KeyBoxSheet {
    const char * k_base = nullptr;
    int64_t n_head_kv = 0;
    int64_t dk = 0;
    int64_t dv = 0;
    int64_t max_blocks = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> tags;
    std::vector<float> boxes;        // [n_head_kv * max_blocks][2 * dk]: mu_k[dk] followed by rad_k[dk]
    std::vector<float> v_centroids;  // [n_head_kv * max_blocks][dv]: mu_v[dv]
    std::vector<float> tail_norms;   // [n_head_kv * max_blocks]: L2 norm of rad_k

    KeyBoxSheet(const char * base, int64_t h_kv, int64_t d_k, int64_t d_v, int64_t blocks)
        : k_base(base), n_head_kv(h_kv), dk(d_k), dv(d_v), max_blocks(blocks),
          tags(new std::atomic<uint64_t>[h_kv * blocks]()),
          boxes(static_cast<size_t>(h_kv * blocks * 2 * d_k), 0.0f),
          v_centroids(static_cast<size_t>(h_kv * blocks * d_v), 0.0f),
          tail_norms(static_cast<size_t>(h_kv * blocks), 0.0f) {}

    const float * v_centroid(int64_t head_kv, int64_t blk_idx, uint64_t expected_tag = 0) const {
        if (blk_idx <= 0 || blk_idx >= max_blocks) { return nullptr; }
        const size_t slot = static_cast<size_t>(head_kv * max_blocks + blk_idx);
        const uint64_t cur_tag = tags[slot].load(std::memory_order_acquire);
        if (cur_tag == 0 || (expected_tag != 0 && cur_tag != expected_tag)) { return nullptr; }
        return v_centroids.data() + slot * static_cast<size_t>(dv);
    }
};

inline KeyBoxSheet * get_box_sheet(const ArgusK1Problem & p, std::mutex & sheet_lock, std::deque<KeyBoxSheet> & box_sheets) {
    const int64_t needed_blocks = std::max<int64_t>(512, (p.n_kv + e1_block - 1) / e1_block);
    std::lock_guard<std::mutex> guard(sheet_lock);
    for (auto & sheet : box_sheets) {
        if (sheet.k_base == p.k && sheet.n_head_kv == p.n_head_kv && sheet.dk == p.dk &&
            sheet.dv == p.dv && sheet.max_blocks >= needed_blocks) {
            return &sheet;
        }
    }
    if (box_sheets.size() >= 64) {
        box_sheets.pop_front();
    }
    box_sheets.emplace_back(p.k, p.n_head_kv, p.dk, p.dv, std::max<int64_t>(needed_blocks, 4096));
    return &box_sheets.back();
}

inline uint64_t fast_block_tag(const void * base, size_t stride_bytes, int64_t n) {
    uint64_t w0 = 0, wn = 0;
    std::memcpy(&w0, base, sizeof(uint64_t));
    std::memcpy(&wn, static_cast<const char *>(base) + (n - 1) * stride_bytes, sizeof(uint64_t));
    const uint64_t h = (w0 ^ ((wn << 13) | (wn >> 51))) * 0x9e3779b97f4a7c15ULL;
    return ((h >> 16) | 1ULL) | (static_cast<uint64_t>(n) << 48);
}

template <typename TK>
float compute_key_box(const TK * keys, size_t stride, int64_t n, int64_t d, float * mu, float * rad) {
    const float inv_n = 1.0f / static_cast<float>(n);
#ifdef ARGUS_K1_SIMD
    const Simd::V inv_n_v = Simd::set1(inv_n);
    if (d % 32 == 0) {
        constexpr int K32 = 32 / Simd::W;
        Simd::V vnorm_sq = Simd::zero();
        for (int64_t i = 0; i < d; i += 32) {
            Simd::V vmin[K32], vmax[K32], vsum[K32], v[K32];
            Simd::load32(keys, i, vmin);
            for (int k = 0; k < K32; ++k) {
                vmax[k] = vmin[k];
                vsum[k] = vmin[k];
            }
            for (int64_t c = 1; c < n; ++c) {
                Simd::load32(keys + c * stride, i, v);
                for (int k = 0; k < K32; ++k) {
                    vmin[k] = Simd::min(vmin[k], v[k]);
                    vmax[k] = Simd::max(vmax[k], v[k]);
                    vsum[k] = Simd::add(vsum[k], v[k]);
                }
            }
            for (int k = 0; k < K32; ++k) {
                const Simd::V m = Simd::mul(vsum[k], inv_n_v);
                const Simd::V r = Simd::max(Simd::sub(vmax[k], m), Simd::sub(m, vmin[k]));
                Simd::store(mu + i + k * Simd::W, m);
                Simd::store(rad + i + k * Simd::W, r);
                vnorm_sq = Simd::fma(r, r, vnorm_sq);
            }
        }
        return std::sqrt(Simd::sum(vnorm_sq));
    }
#endif
    float norm_sq = 0.0f;
    for (int64_t i = 0; i < d; ++i) {
        float lo = element(keys, i), hi = lo, sum = lo;
        for (int64_t c = 1; c < n; ++c) {
            const float v = element(keys + c * stride, i);
            lo   = std::min(lo, v);
            hi   = std::max(hi, v);
            sum += v;
        }
        const float m = sum * inv_n;
        const float r = std::max(hi - m, m - lo);
        mu[i]  = m;
        rad[i] = r;
        norm_sq += r * r;
    }
    return std::sqrt(norm_sq);
}

template <typename TV>
void compute_value_centroid(const TV * values, size_t stride, int64_t n, int64_t dv, float * mu_v) {
    const float inv_n = 1.0f / static_cast<float>(n);
#ifdef ARGUS_K1_SIMD
    const Simd::V inv_n_v = Simd::set1(inv_n);
    if (dv % 32 == 0) {
        constexpr int K32 = 32 / Simd::W;
        for (int64_t i = 0; i < dv; i += 32) {
            Simd::V vsum[K32], v[K32];
            Simd::load32(values, i, vsum);
            for (int64_t c = 1; c < n; ++c) {
                Simd::load32(values + c * stride, i, v);
                for (int k = 0; k < K32; ++k) { vsum[k] = Simd::add(vsum[k], v[k]); }
            }
            for (int k = 0; k < K32; ++k) {
                Simd::store(mu_v + i + k * Simd::W, Simd::mul(vsum[k], inv_n_v));
            }
        }
        return;
    }
#endif
    for (int64_t i = 0; i < dv; ++i) {
        float sum = element(values, i);
        for (int64_t c = 1; c < n; ++c) { sum += element(values + c * stride, i); }
        mu_v[i] = sum * inv_n;
    }
}

// Stage-1a Top-r (32-channel) + Cauchy-Schwarz tail bound or Stage-1b full box bound.
inline std::pair<float, float> box_dot_bounds(const float * q, const QueryTopR & qr, const float * mu,
                                              const float * rad, float block_tail_norm, int64_t d,
                                              float fast_ub_cutoff, bool use_top_r = true) {
    int64_t i = 0;
    float center_dot = 0.0f;
#ifdef ARGUS_K1_SIMD
    Simd::V vc = Simd::zero();
    for (; i + Simd::W <= d; i += Simd::W) {
        vc = Simd::fma(Simd::load(q, i), Simd::load(mu, i), vc);
    }
    center_dot = Simd::sum(vc);
#endif
    for (; i < d; ++i) { center_dot += q[i] * mu[i]; }
    if (fast_ub_cutoff < 1e20f && center_dot > fast_ub_cutoff) {
        return {center_dot, center_dot + block_tail_norm};
    }
    if (use_top_r && d >= 128 && block_tail_norm > 0.0f && fast_ub_cutoff > -1e20f) {
        float l1_top = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const int64_t off = static_cast<int64_t>(qr.segs[k]) * 8;
            for (int j = 0; j < 8; ++j) {
                l1_top += std::fabs(q[off + j] * rad[off + j]);
            }
        }
        const float ub_fast = center_dot + l1_top + qr.q_tail_norm * block_tail_norm;
        return {center_dot, ub_fast};
    }
    int64_t j = 0;
    float l1_rad = 0.0f, l2_rad_sq = 0.0f;
#ifdef ARGUS_K1_SIMD
    Simd::V vl1 = Simd::zero(), vl2 = Simd::zero();
    for (; j + Simd::W <= d; j += Simd::W) {
        const Simd::V qr_v = Simd::abs(Simd::mul(Simd::load(q, j), Simd::load(rad, j)));
        vl1 = Simd::add(vl1, qr_v);
        vl2 = Simd::fma(qr_v, qr_v, vl2);
    }
    l1_rad    = Simd::sum(vl1);
    l2_rad_sq = Simd::sum(vl2);
#endif
    for (; j < d; ++j) {
        const float qr_v = std::fabs(q[j] * rad[j]);
        l1_rad    += qr_v;
        l2_rad_sq += qr_v * qr_v;
    }
    return {center_dot, center_dot + std::min(l1_rad, 1.15f * std::sqrt(l2_rad_sq))};
}

inline void add_scaled_row(float * dst, const float * src, float weight, int64_t d) {
    int64_t i = 0;
#ifdef ARGUS_K1_SIMD
    const Simd::V vw = Simd::set1(weight);
    for (; i + Simd::W <= d; i += Simd::W) {
        Simd::store(dst + i, Simd::fma(Simd::load(src, i), vw, Simd::load(dst, i)));
    }
#endif
    for (; i < d; ++i) { dst[i] += src[i] * weight; }
}

} // namespace argus_k1


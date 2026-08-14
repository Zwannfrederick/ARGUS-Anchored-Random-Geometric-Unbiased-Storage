"""JL operators are cached per (device, dtype, seq_len).

Variable-granularity micro-pages need a differently shaped projection than
full pages, because JL projects along the sequence axis -- a single cached
matrix would be silently wrong for split pages.
"""

import torch

from argus_cache.core.jl_operators import JLOperatorCache


def test_projection_shape_follows_sequence_length():
    cache = JLOperatorCache(page_size=64, ratio=4)

    w = cache.projection(torch.device("cpu"), torch.float32, seq_len=64)

    assert w.shape == (16, 64), "projection must map seq_len -> seq_len/ratio"


def test_different_sequence_lengths_get_different_operators():
    cache = JLOperatorCache(page_size=64, ratio=4)

    full = cache.projection(torch.device("cpu"), torch.float32, seq_len=64)
    micro = cache.projection(torch.device("cpu"), torch.float32, seq_len=16)

    assert full.shape != micro.shape
    assert cache.cache_size() == 2


def test_same_key_returns_the_cached_tensor():
    cache = JLOperatorCache(page_size=64, ratio=4)

    first = cache.projection(torch.device("cpu"), torch.float32, seq_len=64)
    second = cache.projection(torch.device("cpu"), torch.float32, seq_len=64)

    assert first is second, "operator was rebuilt instead of cached"


def test_reconstruction_is_a_right_inverse_of_the_projection():
    """W @ R == I by construction: whatever the prior, the reconstruction must
    reproduce the measurements it was given. If this drifts, the regularized
    solve is wrong and every JL page decodes to something inconsistent with
    its own stored coefficients."""
    cache = JLOperatorCache(page_size=64, ratio=4)
    device, dtype = torch.device("cpu"), torch.float32

    w = cache.projection(device, dtype, seq_len=64)
    recon = cache.reconstruction(device, dtype, seq_len=64)

    identity = w @ recon
    assert identity.shape == (16, 16)
    assert (identity - torch.eye(16)).abs().max() < 1e-3


def test_reconstruction_favours_smooth_sequences_over_noise():
    """The operator's prior is smoothness along the sequence axis (a 1-D
    Laplacian), not low rank. That is the assumption the JL tier makes about
    KV activations, and it is what the tier must be evaluated against --
    a rank-reduced random signal is NOT recovered by this operator.
    """
    cache = JLOperatorCache(page_size=64, ratio=4)
    device, dtype = torch.device("cpu"), torch.float32
    w = cache.projection(device, dtype, seq_len=64)
    recon = cache.reconstruction(device, dtype, seq_len=64)

    def rel_error(x):
        return ((recon @ (w @ x)) - x).norm() / x.norm()

    t = torch.linspace(0, 1, 64, dtype=dtype).unsqueeze(1)
    smooth = torch.cat([torch.sin(2 * torch.pi * k * t) for k in (1, 2, 3)], dim=1)
    noise = torch.randn(64, 3, dtype=dtype)

    smooth_err, noise_err = rel_error(smooth), rel_error(noise)

    assert smooth_err < 0.35, f"smooth signal reconstructed at {smooth_err:.3f}"
    assert noise_err > smooth_err * 2, (
        "the smoothness prior gives no advantage over white noise; the "
        "Laplacian regularization is not doing anything"
    )


def test_clear_empties_the_cache():
    cache = JLOperatorCache(page_size=64, ratio=4)
    cache.projection(torch.device("cpu"), torch.float32, seq_len=64)

    cache.clear()

    assert cache.cache_size() == 0


def test_projection_is_deterministic_across_instances():
    """Two caches must agree, or a page compressed by one and decompressed
    after a restart would decode against a different basis."""
    a = JLOperatorCache(page_size=32).projection(torch.device("cpu"), torch.float32)
    b = JLOperatorCache(page_size=32).projection(torch.device("cpu"), torch.float32)

    assert torch.equal(a, b)


def test_low_rank_input_is_not_recovered_any_better_than_noise():
    """Guards a claim benchmarks/bench_jl_fidelity.py rests on.

    It is tempting to describe this tier as a rank reducer and to justify it
    by the low effective rank of KV activations. That reasoning is wrong: the
    operator carries a smoothness prior, not a low-rank one, and recovers a
    rank-reduced random signal no better than white noise. If this ever stops
    holding, the benchmark's framing needs revisiting -- so it is asserted
    rather than left as a comment.
    """
    cache = JLOperatorCache(page_size=64, ratio=4)
    device, dtype = torch.device("cpu"), torch.float32
    w = cache.projection(device, dtype, seq_len=64)
    recon = cache.reconstruction(device, dtype, seq_len=64)

    def rel_error(x):
        return ((recon @ (w @ x)) - x).norm().item() / x.norm().item()

    torch.manual_seed(0)
    low_rank = torch.randn(64, 4, dtype=dtype) @ torch.randn(4, 8, dtype=dtype)
    white_noise = torch.randn(64, 8, dtype=dtype)

    assert rel_error(low_rank) > 0.8 * rel_error(white_noise), (
        "low-rank input is now recovered much better than noise; the tier may "
        "actually have a low-rank prior after all"
    )


def test_observer_is_notified_only_for_full_page_cuda_operators():
    """The C++ manager keeps a single projection for its CUDA auto-cascade.
    A CPU or micro-page operator must never overwrite it."""
    seen = []
    cache = JLOperatorCache(page_size=64, on_full_page_cuda=lambda kind, t: seen.append(kind))

    cache.projection(torch.device("cpu"), torch.float32, seq_len=64)   # CPU
    cache.projection(torch.device("cpu"), torch.float32, seq_len=16)   # micro

    assert seen == [], "a CPU operator was published to the native manager"

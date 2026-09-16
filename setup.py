"""Builds the CUDA extension. All distribution metadata lives in pyproject.toml.

The extension links against libtorch, so it must be compiled against the same
torch it will later run against; a build-isolated environment would install its
own torch and produce an extension that fails at import with an undefined
symbol. torch therefore stays out of build-system.requires and the documented
install is `pip install --no-build-isolation`.
"""
from setuptools import setup, find_packages

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ModuleNotFoundError as error:
    raise SystemExit(
        "argus_cache builds a PyTorch CUDA extension, so torch must already be\n"
        "importable in the environment that builds it, and it must be the same\n"
        "torch the package will run against.\n\n"
        "    pip install torch\n"
        "    pip install --no-build-isolation argus-cache\n"
    ) from error

setup(
    # Root-level core/ and models/ are re-export shims for this repository's own
    # tests and benchmarks; installing them would put top-level packages by those
    # names into the user's environment.
    packages=find_packages(include=["argus_cache", "argus_cache.*"]),
    ext_modules=[
        CUDAExtension(
            name="argus_cpp_backend",
            sources=[
                "argus_cache/csrc/bindings.cpp",
                "argus_cache/csrc/manager.cpp",
                "argus_cache/csrc/tier_codec.cpp",
                "argus_cache/csrc/zero_copy_pool.cpp",
                "argus_cache/csrc/quantization_kernels.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3"]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)

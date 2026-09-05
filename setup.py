from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="argus_cache",
    version="0.4.0",
    author="Muhammed Emin Çelik",
    description="Heterogeneous KV-cache memory management: paged transformer KV storage across FP16/FP8/INT8/INT4/INT2/1-bit tiers and CPU spill",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=find_packages(),
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
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.0.0",
        "transformers>=4.38.0",
    ],
    extras_require={
        "gateway": ["httpx"],
        "dev": ["pytest", "matplotlib", "httpx"],
    },
    python_requires=">=3.8",
)

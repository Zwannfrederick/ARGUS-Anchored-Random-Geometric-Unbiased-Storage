from setuptools import setup, find_packages

setup(
    name="argus_cache",
    version="0.1.1",
    author="Muhammed Emin Çelik",
    description="ARGUS: Anchored Random Geometric Unbiased Storage - Advanced Dynamic Quantized KV Cache",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.0.0",
        "transformers>=4.38.0",
        "matplotlib",
        "pytest"
    ],
    python_requires=">=3.8",
)

from setuptools import find_packages, setup

setup(
    name="vllm-kv-shrinker",
    version="0.1.0",
    description="RAG-aware KV cache pruning layer for vLLM",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "benchmarks*", "examples*"]),
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
        "vllm": ["vllm>=0.4.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

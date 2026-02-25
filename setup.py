from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cognos-ai",
    version="0.2.0",
    author="Björn Wikström",
    author_email="bjorn@homelab.se",
    description="Epistemological integrity layer for agentic AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bjornshomelab/cognos",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=["numpy>=1.20.0"],
    extras_require={"dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9"]},
)

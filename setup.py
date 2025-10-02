from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ytds",
    version="0.1.0",
    author="YT-DS Team",
    description="Convert YouTube videos to transcribed datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ytds",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "yt-dlp>=2023.3.4",
        "openai>=1.0.0",
        "requests>=2.28.0",
        "pydub>=0.25.1",
        "datasets>=2.10.0",
        "huggingface-hub>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "ytds=ytds.cli:main",
        ],
    },
)

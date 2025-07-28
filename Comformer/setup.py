import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="comformer",
    version="2024.06.05",
    author="Keqiang Yan, Cong Fu, Xiaofeng Qian, Xiaoning Qian, Shuiwang Ji",
    author_email="keqiangyan@tamu.edu",
    description="comformer",
    install_requires=[
            "numpy>=1.19.5, <2",
            "scipy>=1.6.1",
            "jarvis-tools>=2021.07.19",
            "scikit-learn>=0.22.2",
            "matplotlib>=3.4.1",
            "tqdm>=4.60.0",
            "pandas>=1.2.3",
            "pytorch-ignite>=0.4.7",
            "pydantic>=1.8.1,<2",
            "pyparsing>=2.2.1,<3",
            "pandarallel>=1.6.5",
            "e3nn>=0.5.4",
            "nvitop>=1.4.2",
            "pymatgen>=2025.1.24",
            "ipywidgets>=8.1.5",
            "pyarrow>=19.0.0",
            "ipykernel>=6.29.5",    
            "structlog>=25.1.0",
    ],  
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/divelab/AIRS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

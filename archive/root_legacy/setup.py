"""
DeepSequence PWL - Time Series Forecasting with Optional Intermittent Handling
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deepsequence-pwl",
    version="1.0.0",
    author="Mritunjay Kumar",
    author_email="mritunjay.kmr1@gmail.com",
    description="A general-purpose time series forecasting model with optional intermittent demand handling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkuma93/DeepSequence",
    project_urls={
        "Bug Tracker": "https://github.com/mkuma93/DeepSequence/issues",
        "Documentation": "https://github.com/mkuma93/DeepSequence",
        "Source Code": "https://github.com/mkuma93/DeepSequence",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "jupyter>=1.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

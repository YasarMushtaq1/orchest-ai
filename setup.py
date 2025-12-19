"""
Setup script for OrchestAI
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if it exists
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split("\n")
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="orchestai",
    version="0.1.0",
    author="Sayed Yasar Ahmad Mushtaq",
    author_email="271mushtaq@gmail.com",
    description="Autonomous Multi-Model Orchestration via Learned Task Planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YasarMushtaq1/orchest-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "orchestai-train-phase1=scripts.train_phase1:main",
            "orchestai-train-phase2=scripts.train_phase2:main",
            "orchestai-evaluate=scripts.evaluate:main",
            "orchestai-inference=scripts.run_inference:main",
        ],
    },
)


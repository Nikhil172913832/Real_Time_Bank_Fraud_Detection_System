"""
Setup configuration for Real-Time Bank Fraud Detection System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="fraud-detection-system",
    version="1.2.0",
    author="Nikhil",
    author_email="nikhil.dev@example.com",
    description="Production-grade real-time bank fraud detection system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nikhil172913832/Real_Time_Bank_Fraud_Detection_System",
    project_urls={
        "Bug Tracker": "https://github.com/Nikhil172913832/Real_Time_Bank_Fraud_Detection_System/issues",
        "Documentation": "https://github.com/Nikhil172913832/Real_Time_Bank_Fraud_Detection_System/blob/main/README.md",
        "Source Code": "https://github.com/Nikhil172913832/Real_Time_Bank_Fraud_Detection_System",
    },
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=24.1.1",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "pylint>=3.0.3",
            "isort>=5.13.2",
            "pre-commit>=3.6.0",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=2.0.0",
            "sphinx-autodoc-typehints>=1.25.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "fraud-api=app:main",
            "fraud-inference=inference:main",
            "fraud-dashboard=dashboard:main",
            "fraud-train=training:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

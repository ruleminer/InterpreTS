from setuptools import setup, find_packages

setup(
    name="InterpreTS",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0"
    ],
    description="Feature extraction from time series to support the creation of interpretable and explainable predictive models.",
    long_description=open("docs/README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/ruleminer/InterpreTS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    keywords="time series feature extraction interpretability explainability machine learning",
    project_urls={
        "Documentation": "https://github.com/ruleminer/InterpreTS/docs",
        "Source": "https://github.com/ruleminer/InterpreTS",
        "Tracker": "https://github.com/ruleminer/InterpreTS/issues",
    },
)

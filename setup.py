from setuptools import setup, find_packages

setup(
    name="interpreTS",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "sktime>=0.34.0"
    ],
    description="Feature extraction from time series to support the creation of interpretable and explainable predictive models.",
    long_description=open("docs/README.md").read(),
    long_description_content_type="text/markdown",
    author=['Veragath', 'MartynaZur', 'martyna-kramarz', 'JaykerX', 'piotrek1459', 'xTaromarux'],
    url="https://github.com/ruleminer/InterpreTS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    keywords="time series feature extraction interpretability explainability machine learning",
    project_urls={
        "Documentation": "https://github.com/ruleminer/InterpreTS/docs",
        "Source": "https://github.com/ruleminer/InterpreTS",
        "Tracker": "https://github.com/ruleminer/InterpreTS/issues",
    },
)

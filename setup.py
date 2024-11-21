from setuptools import setup, find_packages

setup(
    name="InterpreTS",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy",
        "statsmodels",
    ],
    description="Feature extraction from time series to support the creation of interpretable and explainable predictive models.",
    long_description=open("docs/README.md").read(),
    long_description_content_type="text/markdown",
    author=["Łukasz Wróbel", "Sławomir Put", "Martyna Żur", "Martyna Kramarz", "Jarosław Strzelczyk", "Weronika Wołowczyk", "Piotr Krupiński"],
    url="https://github.com/ruleminer/InterpreTS",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    test_suite="tests",
    keywords="time series feature extraction interpretability explainability machine learning",
    project_urls={
        "Documentation": "https://github.com/ruleminer/InterpreTS/docs",
        "Source": "https://github.com/ruleminer/InterpreTS",
        "Tracker": "https://github.com/ruleminer/InterpreTS/issues",
    },
)

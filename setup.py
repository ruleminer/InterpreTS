from setuptools import setup, find_packages

setup(
    name="InterpreTS",
    version="0.4.1",
    packages=find_packages(),
    install_requires=[
<<<<<<< HEAD
        "pandas==2.1.2",
        "numpy==1.26.1",
        "statsmodels==0.14.0",
        "langchain_community==0.0.17",
        "langchain==0.1.5",
        "openai==0.28.0",
        "streamlit==1.26.0",
=======
        "pandas==2.2.3",
        "numpy==2.2.1",
        "statsmodels==0.14.4",
        "langchain_community==0.3.14",
        "langchain==0.3.14",
        "openai==1.59.4",
        "streamlit==1.41.1",
        "scikit-learn==1.6.0",
>>>>>>> development
        "joblib==1.4.2",
        "tqdm==4.67.1",
        "dask==2024.12.1",
        "scipy==1.15.0",
        "pillow==11.1.0"
    ],
    
    description="Feature extraction from time series to support the creation of interpretable and explainable predictive models.",
    long_description=open("docs/README.md").read(),
    long_description_content_type="text/markdown",
    author=["Sławomir Put", "Martyna Żur", "Weronika Wołowczyk", "Jarosław Strzelczyk", "Piotr Krupiński", "Martyna Kramarz", "Łukasz Wróbel"],
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

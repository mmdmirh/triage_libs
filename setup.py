from setuptools import setup, find_packages

setup(
    name="triage_libs",
    version="0.1.27",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "openpyxl",
        "xlrd>=2.0.1",
        "holidays",
        "msoffcrypto-tool",
        "statsmodels",
        "scipy",
        "matplotlib",
        "seaborn",
    ],
    author="Mohamad",
    author_email="mirhos5@mcmaster.ca",
    description="Triage data preprocessing and loading libraries",
    url="https://mmdmirh.github.io/triage_libs/",
)

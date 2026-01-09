from setuptools import setup, find_packages

setup(
    name="triage_libs",
    version="0.1.11",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "openpyxl",
        "xlrd>=2.0.1",
        "holidays",
    ],
    author="Mohamad",
    description="Triage data preprocessing and loading libraries",
)

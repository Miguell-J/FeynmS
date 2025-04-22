from setuptools import setup, find_packages

setup(
    name="FeynmS",
    version="0.1.4",
    description="A Python library for quantum computing simulation.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Miguel Araújo Julio",
    author_email="Julioaraujo.guel@gmail.com",
    url="https://github.com/Miguell-J/FeynmS",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19",
        "scipy>=1.7",
        "matplotlib>=3.4",
        "qiskit>=0.34"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    license="MIT",
)

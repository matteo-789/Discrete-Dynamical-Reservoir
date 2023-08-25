import os

from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="Discrete_Dynamical_Reservoir",
        version="0.1.4",
        author="Matteo Cisneros",
        description="Use of discrete dynamical systems within recurrent neural networks",
        packages=find_packages(),
        classifiers=[
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        python_requires=">=3.6",
    )

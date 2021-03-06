import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="leg_controllers",
    version="0.0.1",
    author="Joseph Sullivan",
    author_email="jgs6156@uw.edu",
    description="Control routines for the pareto leg project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sburden-group/pareto-leg-control",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
)
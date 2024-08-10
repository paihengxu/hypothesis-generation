import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="hypogenic",
    version="0.0.1",
    author="HypoGenic",
    author_email="",
    description="A package for generating and evaluating hypotheses.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": ["hypogenic_generation=hypogenic_cmd.generation:main"],
    },
)

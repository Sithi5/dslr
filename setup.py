from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []

test_requirements = ["pytest==6.2.5"]

dev_requirements = ["black==21.10b0"] + test_requirements

extra_requirements = {
    "dev": dev_requirements,
}


setup(
    name="dslr",
    version="0.0.1",
    description="Datascience X Logistic Regression.",
    long_description=long_description,
    author="Malo Boucé - Mathieu Ginisty",
    author_email="ma.sithis@gmail.com",
    url="https://github.com/Sithi5/dslr",
    packages=["src", "tests"],
    install_requires=requirements,
    extras_require=extra_requirements,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

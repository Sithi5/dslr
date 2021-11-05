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
    author="Malo Boucé, Mathieu Ginisty",
    author_email="ma.sithis@gmail.com",
    url="https://github.com/Sithi5/dslr",
    packages=["tests", "scripts", "predict", "train"],
    entry_points={
        "console_scripts": [
            "logreg_predict = predict.logreg_predict:cli",
            "logreg_train = train.logreg_train:cli",
            # SCRIPTS
            "describe = scripts.describe:cli",
            "histogram = scripts.histogram:cli",
            "pair_plot = scripts.pair_plot:cli",
            "scatter_plot = scripts.scatter_plot:cli",
            # END OF SCRIPTS
        ],
    },
    install_requires=requirements,
    extras_require=extra_requirements,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

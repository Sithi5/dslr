from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

requirements = [
    "pandas==1.4.4",
    "numpy==1.23.3",
    "progress==1.6",
    "matplotlib==3.5.1",
    "seaborn==0.11.2",
    "colorama==0.4.5",
    "logger-1.0.6 @ git+https://github.com/Sithi5/logger@1.0.6",
]

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
    packages=["tests", "dslr"],
    entry_points={
        "console_scripts": [
            "logreg_predict = dslr.predict.logreg_predict:cli",
            "logreg_train = dslr.train.logreg_train:cli",
            # SCRIPTS
            "describe = dslr.scripts.describe:cli",
            "histogram = dslr.scripts.histogram:cli",
            "pair_plot = dslr.scripts.pair_plot:cli",
            "scatter_plot = dslr.scripts.scatter_plot:cli",
            ## BONUS
            "check = dslr.scripts.check_results:cli",
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

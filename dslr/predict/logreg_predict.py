import argparse

import numpy as np

from dslr.scripts.utils import open_datafile, standardize, sigmoid
from dslr.train.logger import Logger


def cli():
    parser = argparse.ArgumentParser(
        description="Logistic regression training program for DataScience X Logistic \
        Regression project. "
    )
    parser.add_argument("dataset", type=open_datafile, help="input a csv file.")
    parser.add_argument(
        "-l",
        "--level",
        metavar="log-level",
        choices=["ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        default="INFO",
    )
    args = parser.parse_args()
    logger = Logger(level=args.level)


if __name__ == "__main__":
    cli()

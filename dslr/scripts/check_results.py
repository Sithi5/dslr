import argparse
import pandas as pd
from dslr.utils import open_datafile
from logger.Logger import Logger


def test_accuracy(dataset1, dataset2, logger):
    score = 0
    for i in range(len(dataset1)):
        if dataset1[i] == dataset2[i]:
            score += 1
        else:
            logger.debug(f"Diff line {i + 2}, index = {i}\n'{dataset1[i]}' vs '{dataset2[i]}'")
    logger.info(f"\nAccuracy: {score / len(dataset1):.2f}")


def cli():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset1", type=open_datafile, help="First dataset to compare.")
    parser.add_argument("dataset", type=open_datafile, help="Second dataset to compare.")
    parser.add_argument(
        "-l",
        "--level",
        metavar="log-level",
        choices=["ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Choices: ERROR, WARNING, INFO, DEBUG. The parameter set by default is INFO.",
    )
    args = parser.parse_args()
    logger = Logger(level=args.level, display_levels=False)

    try:
        dataset1 = args.dataset1.loc[:, "Hogwarts House"]
        dataset2 = args.dataset.loc[:, "Hogwarts House"]
        test_accuracy(dataset1, dataset2, logger)
    except Exception as error:
        print("Something went wrong:", error)


if __name__ == "__main__":
    cli()

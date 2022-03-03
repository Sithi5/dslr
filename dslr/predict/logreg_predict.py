import argparse
import numpy as np
import matplotlib.pyplot as plt

from colorama import Fore, Style
from dslr.utils import open_datafile, sigmoid, create_csv
from dslr.train.logger import Logger
from dslr.train.logreg_train import house, house_rev


def is_valid(df):
    df = df[["Hogwarts House"]]
    if df.isnull().values.any():
        return 0
    return 1


def pie_chart(results, title, n):
    labels = []
    sizes = []
    for i in results:
        if i not in labels:
            labels.append(i)
            sizes.append(0)
    for i in results:
        sizes[labels.index(i)] += 1
    plt.figure(n)
    plt.title(title)
    plt.pie(sizes, labels=labels, autopct="%1.2f%%", shadow=True, startangle=90)


def standardize_predict(x, mean_1, mean_2, std_1, std_2):
    mean = np.array([mean_1, mean_2])
    std = np.array([std_1, std_2])
    x = (x - mean) / std
    x = np.insert(x, 0, 1, axis=0)
    return x


def print_piechart(csv_list, df):
    student_result = np.array(csv_list)[1:, 1]
    pie_chart(student_result, "Piechart for students", 0)
    if is_valid(df):
        pie_chart(df["Hogwarts House"].to_list(), "Actual data", 1)
    plt.show()


def predict(df, weights, logger):
    csv_list = [["Index", "Hogwarts House"]]
    for i in range(len(df)):
        result = [[], [], [], []]
        student = df.loc[i]
        for row in weights:
            if not np.isnan(student[row[1]]) and not np.isnan(student[row[2]]):
                features = np.array([student[row[1]], student[row[2]]])
                thetas = np.array([row[4], row[5], row[6]])
                features = standardize_predict(features, row[7], row[8], row[9], row[10])
                result[house[row[0]] - 1].append(sigmoid(np.dot(features, thetas)))
        for j in range(0, 4):
            if len(result) != 0:
                result[j] = sum(result[j]) / len(result[j])
            else:
                result[j] = 0
        maximum = max(result)
        index = result.index(maximum)
        rounded = [round(result[l], 2) for l in range(4)]
        logger.debug(
            f"{Fore.YELLOW}{rounded},{Style.RESET_ALL} best = {Fore.GREEN}{result[index]:.2f}{Style.RESET_ALL}, house for best = {Fore.BLUE}{house_rev[index + 1]}{Style.RESET_ALL}"
        )
        csv_list.append([i, house_rev[index + 1]])
    return csv_list


def cli():
    parser = argparse.ArgumentParser(
        description="Logistic regression predict program for DataScience X Logistic \
        Regression project. "
    )
    parser.add_argument("dataset", type=open_datafile, help="input a csv file with the dataset.")
    parser.add_argument("weights", type=open_datafile, help="input a csv file with the weights.")
    parser.add_argument(
        "-l",
        "--level",
        metavar="log-level",
        choices=["ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        default="INFO",
    )
    parser.add_argument(
        "-p", "--piechart", action="store_true", help="print a piechart for the results"
    )
    args = parser.parse_args()

    logger = Logger(level=args.level, name="Log from logreg_predict.py")
    df = args.dataset
    try:
        weights = args.weights.to_numpy()
        df.drop(
            [
                "Index",
                "Arithmancy",
                "Flying",
                "Care of Magical Creatures",
                "Potions",
                "Transfiguration",
            ],
            axis=1,
            inplace=True,
        )
        df = df[["Hogwarts House"] + list(df.select_dtypes(include="number").columns)]

        csv_list = predict(df, weights, logger)
        if args.piechart:
            print_piechart(csv_list, df)
        create_csv(csv_list, "houses.csv")
        logger.info("houses.csv created. index is in dataset for student place.")
    except Exception as error:
        print("Something went wrong : ", error)


if __name__ == "__main__":
    cli()

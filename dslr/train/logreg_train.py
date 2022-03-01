import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np

from dslr.scripts.utils import open_datafile, standardize, sigmoid
from dslr.train.logger import Logger

house = {"Ravenclaw": 1, "Slytherin": 2, "Gryffindor": 3, "Hufflepuff": 4}
house_rev = {value: key for key, value in house.items()}


def clean_data(df, f1, f2, house):
    x, y = [], []
    df = df.to_numpy()
    for row in df:
        if not np.isnan(row[f1]) and not np.isnan(row[f2]):
            x.append([row[f1], row[f2]])
            y.append(1 if row[0] == house else 0)
    return np.array(x), np.array(y)


def get_accuracy(x, y, thetas):
    length = len(x)
    correct = 0
    for i in range(length):
        prediction = 1 if sigmoid(x[i].dot(thetas)) >= 0.5 else 0
        correct = correct + 1 if y[i] == prediction else correct
    return (correct / length) * 100


def create_csv(row_list, name):
    with open(name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def display_data(x, y, house, df, f1, f2, thetas=None):
    plt.figure()
    if thetas is not None:
        x_value = np.array([np.min(x[:, 1]), np.max(x[:, 1])])
        y_value = -(thetas[0] + thetas[1] * x_value) / thetas[2]
        plt.plot(x_value, y_value, "g")
        pos, neg = (y == 1).reshape(len(x[:, 0]), 1), (y == 0).reshape(len(x[:, 2]), 1)
        plt.scatter(x[pos[:, 0], 1], x[pos[:, 0], 2], c="r", marker="+")
        plt.scatter(x[neg[:, 0], 1], x[neg[:, 0], 2], marker="o", s=10)
    else:
        pos, neg = (y == 1).reshape(len(x[:, 0]), 1), (y == 0).reshape(len(x[:, 1]), 1)
        plt.scatter(x[pos[:, 0], 0], x[pos[:, 0], 1], c="r", marker="+")
        plt.scatter(x[neg[:, 0], 0], x[neg[:, 0], 1], marker="o", s=10)

    plt.xlabel(df.columns[f1])
    plt.ylabel(df.columns[f2])
    plt.legend([house, "Not " + house], loc=0)
    plt.title(f"{df.columns[f1]} vs {df.columns[f2]} data")
    plt.show()


def train_thetas(x, y, thetas, lr, epochs):
    pass


def training(args):
    logger = Logger(level=args.level)

    df = args.dataset
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

    csv_list = [["House", "F1", "F2", "Accuracy", "T0", "T1", "T2"]]

    for i in range(1, 5):
        if args.level == "INFO":
            logger("")
        for f1 in range(1, len(df.columns) - 1):
            for f2 in range(f1 + 1, len(df.columns) - 1):
                x, y = clean_data(df, f1, f2, house_rev[i])
                x = standardize(x)

                row, col = x.shape[0], x.shape[1]
                y = y.reshape(row, 1)

                thetas = np.zeros((col + 1, 1))
                thetas = train_thetas(x, y, thetas, args.lr, args.epochs)

                accuracy = get_accuracy(x, y, thetas)
                if accuracy >= args.accuracy:
                    csv_list.append(
                        [
                            house_rev[i],
                            df.columns[f1],
                            df.columns[f2],
                            accuracy,
                            thetas[0][0],
                            thetas[1][0],
                            thetas[2][0],
                        ]
                    )
    create_csv(csv_list, "info.csv")


def cli():
    parser = argparse.ArgumentParser(
        description="Logistic regression training program for DataScience X Logistic \
        Regression project. "
    )
    parser.add_argument("dataset", type=open_datafile, help="input a csv file.")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="Epochs")
    parser.add_argument(
        "-l",
        "--level",
        metavar="log-level",
        choices=["ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        default="INFO",
    )
    parser.add_argument("-l", "--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("-a", "--accuracy", type=int, default=97, help="Minimal accuracy")
    args = parser.parse_args()
    training(args)


if __name__ == "__main__":
    cli()

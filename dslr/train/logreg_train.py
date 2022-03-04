import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np

from colorama import Fore, Style
from dslr.utils import open_datafile, standardize, sigmoid, create_csv
from dslr.train.logger import Logger
from progress.bar import ChargingBar

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


def display_loss(loss, f1, f2, df):
    plt.figure()
    plt.plot(range(len(loss)), loss)
    plt.ylabel("Loss")
    plt.xlabel("epochs")
    plt.title(f"Loss graph of {df.columns[f1]} vs {df.columns[f2]}")
    plt.show()


def cost_function(x, y, thetas):
    m = len(x)
    h0 = sigmoid(np.dot(x, thetas))
    return -(1 / m) * np.sum(y * np.log(h0) + (1 - y) * np.log(1 - h0))


def gradient(x, y, thetas):
    m = len(x)
    h0 = sigmoid(np.dot(x, thetas))
    dw = (1 / m) * np.dot(x.T, (h0 - y))
    return dw


def train_thetas(x, y, thetas, lr, epochs):
    loss = []
    for _ in range(epochs):
        dw = gradient(x, y, thetas)
        thetas -= lr * dw
        loss.append(cost_function(x, y, thetas))
    return loss, thetas


def training(args, logger):
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
    csv_list = [
        [
            "House",
            "F1",
            "F2",
            "Accuracy",
            "T0",
            "T1",
            "T2",
            "mean_F1",
            "mean_F2",
            "std_F1",
            "std_F2",
        ]
    ]

    if args.level != "DEBUG":
        bar = ChargingBar(
            "Training in progress: ",
            max=((len(df.columns) - 1) * (len(df.columns) - 2) * 1.4),
            suffix="%(percent)d%%",
        )

    for i in range(1, 5):
        logger.debug(f"{Fore.YELLOW}{house_rev[i].upper()}{Style.RESET_ALL}")
        for f1 in range(1, len(df.columns) - 1):
            for f2 in range(f1 + 1, len(df.columns) - 1):
                x, y = clean_data(df, f1, f2, house_rev[i])
                if args.show:
                    display_data(x, y, house_rev[i], df, f1, f2)
                x, mean, std = standardize(x)

                row, col = x.shape[0], x.shape[1]
                y = y.reshape(row, 1)
                x = np.insert(x, 0, 1, axis=1)

                thetas = np.zeros((col + 1, 1))
                loss, thetas = train_thetas(x, y, thetas, args.lr, args.epochs)

                accuracy = get_accuracy(x, y, thetas)
                line = Fore.GREEN
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
                            mean[0],
                            mean[1],
                            std[0],
                            std[1],
                        ]
                    )
                    if args.show:
                        display_data(x, y, house_rev[i], df, f1, f2, thetas)
                    if args.loss:
                        display_loss(loss, f1, f2, df)
                else:
                    line = Fore.RED
                line += f"{df.columns[f1]:29s} vs {df.columns[f2]:29s} : Accuracy = {accuracy:.2f}%{Style.RESET_ALL}"
                logger.debug(line)

                if args.level != "DEBUG":
                    bar.next()

    create_csv(csv_list, "weights.csv")
    if args.level != "DEBUG":
        bar.finish()
    logger.info("weights.csv created, you can use it on predict program.")


def cli():
    parser = argparse.ArgumentParser(
        description="Logistic regression training program for DataScience X Logistic \
        Regression project. "
    )
    parser.add_argument("dataset", type=open_datafile, help="input a csv file.")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="Epochs")
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        default=False,
        help="Show some graphic for comprehension.",
    )
    parser.add_argument(
        "-lo",
        "--loss",
        action="store_true",
        default=False,
        help="Show graphic function of loss for each versus of features.",
    )
    parser.add_argument(
        "-l",
        "--level",
        metavar="log-level",
        choices=["ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        default="INFO",
        help="Choices: ERROR, WARNING, INFO, DEBUG. The parameter set by default is INFO.",
    )
    parser.add_argument("-lr", "--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument(
        "-a", "--accuracy", type=int, default=97, help="Minimal accuracy, Set to 97 by default."
    )
    args = parser.parse_args()
    logger = Logger(level=args.level, name="Log from logreg_train.py")

    if args.level == "DEBUG":
        training(args, logger)
    else:
        try:
            training(args, logger)
        except Exception as error:
            print("Something went wrong : ", error)


if __name__ == "__main__":
    cli()

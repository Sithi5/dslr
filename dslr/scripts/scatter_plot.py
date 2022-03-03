import argparse
import matplotlib.pyplot as plt

from dslr.utils import open_datafile, get_key
from dslr.train.logreg_train import house, house_rev


def scatter_plot(data, col1, col2):
    plt.figure()
    plt.xlabel(col1)
    plt.ylabel(col2)
    for house in range(1, 5):
        x = []
        y = []
        for index, row in data.iterrows():
            if int(row["Hogwarts House"]) == house:
                x.append(row[col1])
                y.append(row[col2])
        plt.scatter(x, y, marker=".")
    plt.show()


def cli():
    parser = argparse.ArgumentParser(
        description="DataScience X Logistic \
        Regression program"
    )
    parser.add_argument("dataset", type=open_datafile, help="input a csv file.")
    parser.add_argument("feature_1", type=str, nargs="?", default="Astronomy")
    parser.add_argument("feature_2", type=str, nargs="?", default="Defense Against the Dark Arts")
    args = parser.parse_args()
    args.dataset["Hogwarts House"].replace(house, inplace=True)
    data = args.dataset.select_dtypes("number").dropna()
    scatter_plot(data, args.feature_1, args.feature_2)


if __name__ == "__main__":
    cli()

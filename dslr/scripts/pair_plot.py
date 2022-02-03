import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from dslr.scripts.utils import open_datafile


def pair_plot(data):
    sns.set(style="whitegrid", color_codes=True)
    sns.pairplot(data, hue="Hogwarts House", markers=".")
    plt.show()


def cli():
    parser = argparse.ArgumentParser(
        description="DataScience X Logistic \
        Regression program"
    )
    parser.add_argument("dataset", type=open_datafile, help="input a csv file.")
    args = parser.parse_args()
    args.dataset.drop("Index", axis=1, inplace=True)
    data = args.dataset[
        ["Hogwarts House"] + list(args.dataset.select_dtypes("number").columns)
    ].dropna()
    print(data)
    pair_plot(data)


if __name__ == "__main__":
    cli()

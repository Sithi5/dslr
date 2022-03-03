import argparse
import matplotlib.pyplot as plt

from dslr.utils import get_key, open_datafile, std


houses = {"Gryffindor": 1, "Hufflepuff": 2, "Ravenclaw": 3, "Slytherin": 4}


def display_histogram(col, data):
    plt.figure()
    plt.title(col)
    for i in range(1, 5):
        current = []
        for index, elem in enumerate(data[col]):
            if data["Hogwarts House"][index] == i:
                current.append(elem)
        house = get_key(i, houses)
        plt.hist(current, alpha=0.5, label=house)
        plt.ylabel("Frequency")
        plt.legend(loc="upper left")
    plt.show()


def histogram(dataset):
    dataset["Hogwarts House"].replace(houses, inplace=True)
    data = dataset.select_dtypes("number")
    min_std = 9999999
    best = ""
    for col in data.columns:
        if col != "Hogwarts House" and col != "Index":
            if min_std > std(list(data[col])):
                min_std = std(list(data[col]))
                best = col
    display_histogram(best, data)


def cli():
    parser = argparse.ArgumentParser(
        description="DataScience X Logistic \
        Regression program"
    )
    parser.add_argument("dataset", type=open_datafile, help="input a csv file.")
    args = parser.parse_args()
    histogram(args.dataset)


if __name__ == "__main__":
    cli()

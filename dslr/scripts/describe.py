import argparse
import pandas as pd

from dslr.utils import (
    count,
    mean,
    std,
    min,
    quarter,
    median,
    three_quarters,
    max,
    unique,
    freq,
    open_datafile,
)


functions_dict = {
    0: count,
    1: mean,
    2: std,
    3: min,
    4: quarter,
    5: median,
    6: three_quarters,
    7: max,
    8: unique,
    9: freq,
}


def describe_display(rows, data):
    cols = list(data.columns)
    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    print(df)


def describe_dataset(dataset):
    rows = {
        "Count": [],
        "Mean": [],
        "Std": [],
        "Min": [],
        "25%": [],
        "50%": [],
        "75%": [],
        "Max": [],
        "Unique": [],
        "Freq": [],
    }
    cols_empty = [col for col in dataset.columns if dataset[col].isnull().all()]
    dataset.drop(cols_empty, axis=1, inplace=True)
    data = dataset.select_dtypes("float64")
    for col in data.columns:
        lister = list(data[col])
        for i, elem in enumerate(rows):
            rows[elem].append(functions_dict[i](lister))
    describe_display(rows, data)


def cli():
    parser = argparse.ArgumentParser(description="DataScience Describe program.")
    parser.add_argument("dataset", type=open_datafile, help="input a csv file.")
    args = parser.parse_args()
    try:
        describe_dataset(args.dataset)
    except Exception as error:
        print("Something went wrong : ", error)


if __name__ == "__main__":
    cli()

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
    }
    cols_vides = [col for col in dataset.columns if dataset[col].isnull().all()]
    dataset.drop(cols_vides, axis=1, inplace=True)
    data = dataset.select_dtypes("float64")
    for col in data.columns:
        lister = list(data[col])
        for i, elem in enumerate(rows):
            rows[elem].append(functions_dict[i](lister))
    describe_display(rows, data)


def cli():
    parser = argparse.ArgumentParser(
        description="DataScience X Logistic \
        Regression program"
    )
    parser.add_argument("dataset", type=open_datafile, help="input a csv file.")
    args = parser.parse_args()
    describe_dataset(args.dataset)


if __name__ == "__main__":
    cli()

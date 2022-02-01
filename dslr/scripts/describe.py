import argparse
from asyncio.windows_events import NULL
import numpy as np
import pandas as pd
from dslr.scripts.utils import count, mean, std, min, quarter, median, three_quarters, max


functions_dict = {
    0 : count,
    1 : mean,
    2 : std,
    3 : min,
    4 : quarter,
    5 : median,
    6 : three_quarters,
    7 : max
}


def describe_display(rows, data):
    cols = list(data.columns)
    df = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    print(df)


def describe_dataset(dataset):
    rows = {
    "Count": [],
    "Mean" : [],
    "Std": [],
    "Min": [],
    "25%": [],
    "50%": [],
    "75%": [],
    "Max": []
    }
    data = dataset.select_dtypes("float64")
    for col in data.columns:
        lister = list(data[col])
        for i, elem in enumerate(rows):
            rows[elem].append(functions_dict[i](lister))
    describe_display(rows, data)

def open_datafile(datafile):
    try:
        data = pd.read_csv(datafile)
    except pd.errors.EmptyDataError:
        exit("Empty data file.")
    except pd.errors.ParserError:
        raise argparse.ArgumentTypeError("Error parsing file, needs to be a \
            well formated csv.")
    except Exception as error:
        exit(f"{error}: File {datafile} corrupted or does not exist.")
    return data


def cli():
    parser = argparse.ArgumentParser(description="DataScience X Logistic \
        Regression program")
    parser.add_argument("dataset", type=open_datafile, help="input a csv file.")
    args = parser.parse_args()
    describe_dataset(args.dataset)


if __name__ == "__main__":
    cli()

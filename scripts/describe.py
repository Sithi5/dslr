import argparse
import numpy as np
import pandas as pd

def describe_dataset(dataset):
    data = dataset.select_dtypes('float64')
    rows = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    column = []
    for i in data:
        column.append(i)
    

def open_datafile(datafile):
    try:
        data = pd.read_csv(datafile)
    except pd.errors.EmptyDataError:
        exit("Empty data file.")
    except pd.errors.ParserError:
        raise argparse.ArgumentTypeError("Error parsing file, needs to be a well formated csv.")
    except Exception as error:
        exit(f"{error}: File {datafile} corrupted or does not exist.")
    return data


def cli():
    parser = argparse.ArgumentParser(description="DataScience X Logistic Regression program")
    parser.add_argument("dataset", type=open_datafile, help="input a csv file.")
    args = parser.parse_args()
    describe_dataset(args.dataset)


if __name__ == "__main__":
    cli()

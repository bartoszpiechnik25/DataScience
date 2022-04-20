import pandas as pd
from blueprints import plot_missing_data
import sys


class LoadData:
    """
    Reads your csv data from specified path and load it to
    pandas.DataFrame object.
    ---------------------------------------------------------
    If path does not exist, or you were given invalid name it
    will return -1 and stop reading process.
    ---------------------------------------------------------"""
    def __init__(self, path):

        try:
            self.path = path
            self.df = pd.read_csv(path)
        except:
            print(f'Specified path -> {self.path} does not exists')
            sys.exit(-1)

    def create_data(self):
        """
    This function will summarize your dataset and display:
    ======================================================
    info: df.info()
    shape: df.shape
    head: df.head()
    describe: df.describe()
    number of NaN values: df.isna().sum()
    plot number of NaN on heatmap: sns.heatmap()
    number of duplicates: df.duplicated().sum()
    ======================================================"""
        print(f'{self.df.info()}\nShape of DataFrame is {self.df.shape}\n')
        print(f'Head of DataFrame:\n{self.df.head()}')
        print(f'Dataset statistics:\n{self.df.describe()}')
        print(f'Number of NaN values in DataFrame:\n{self.df.isna().sum()}')
        plot_missing_data(self.df)
        print(f'Number of duplicates in DataFrame: {self.df.duplicated().sum()}')
        return self.df


if __name__ == '__main__':
    load = LoadData('titanic_train.csv')
    df = load.create_data()
    df.head()
    print(LoadData.create_data.__doc__)
    print(LoadData.__doc__)



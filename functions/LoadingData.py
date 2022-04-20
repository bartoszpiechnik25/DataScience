import pandas as pd
from blueprints import plot_missing_data


class LoadData:
    def __init__(self, path):
        try:
            self.df = pd.read_csv(path)
            self.path = path
        except:
            print(f'Specified path -> {self.path} does not exists')

    def create_data(self):
        print(f'{self.df.info()}\n Shape of DataFrame is {self.df.shape}\n')
        print(f'Head of DataFrame:\n{self.df.head()}')
        print(f'Dataset statistics:\n{self.df.describe()}')
        print(f'Number of NaN values in DataFrame:\n{self.df.isna().sum()}')
        plot_missing_data(self.df)
        print(f'Number of duplicates in DataFrame: {self.df.duplicated().sum()}')
        return self.df



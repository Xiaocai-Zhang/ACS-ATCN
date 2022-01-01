import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np


class FormatData:
    def __init__(self,train_inp,train_oup):
        self.train_inp=train_inp
        self.train_oup=train_oup

    def ToArray(self,Df):
        Array = Df.to_numpy(dtype=np.float16)
        return Array

    def main(self):
        train_inp = self.ToArray(self.train_inp)
        train_inp = np.expand_dims(train_inp, axis=2)

        train_oup = self.ToArray(self.train_oup)

        return train_inp,train_oup

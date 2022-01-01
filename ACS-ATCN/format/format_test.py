import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np



class FormatData:
    def __init__(self,test_inp,test_oup):
        self.test_inp=test_inp
        self.test_oup=test_oup

    def ToArray(self,Df):
        Array = Df.to_numpy(dtype=np.float16)
        return Array

    def main(self):
        test_inp = self.ToArray(self.test_inp)
        test_inp = np.expand_dims(test_inp, axis=2)

        test_oup = self.ToArray(self.test_oup)

        return test_inp,test_oup

import pandas as pd
pd.options.mode.chained_assignment = None
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from format.format_test import FormatData
from evaluate.evaluation_test import Evaluate
os.chdir(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = ""



class config:
    '''
    configure parameters and paths
    '''
    n_classes = 2
    TrainValRatio = [0.8, 0.2]
    TestInpPath = '../../data/proced/HandOutlines/test_input.csv'
    TestOupPath = '../../data/proced/HandOutlines/test_target.csv'
    ModelPath = '../../save_model/HandOutlines/'
    ParaPath = '../../para/HandOutlines/parameter.csv'
    SavePath = '../../para/HandOutlines/test_result.csv'


class attention(Layer):
    '''
    attention mechanism
    '''
    def __init__(self, return_sequences=True,**kwargs):
        self.return_sequences = return_sequences
        super(attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences
        })
        return config


def Test(x,round_no):
    '''
    prediction based on the best model
    '''
    batchsize = int(round(x[2]))
    timestep = int(round(x[3]))

    # mix precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # read data
    test_inp = pd.read_csv(config.TestInpPath, header=None)
    test_oup = pd.read_csv(config.TestOupPath, header=None)

    # process data, raw data alread normalized
    FDClass = FormatData(test_inp, test_oup)
    test_inp, test_oup = FDClass.main()

    LastIdx = [idx for idx in range(2709)]
    ValdIdx = LastIdx[-timestep:]

    test_inp = test_inp[:, ValdIdx, :]

    string = ""
    for i in range(len(x)):
        if i in [2, 3, 4, 5, 6]:
            item = int(round(x[i]))
        else:
            item = round(x[i], 4)
        string = string + '_' + str(item)

    SaveModlPath = config.ModelPath + 'round_' + str(round_no) + '/'
    SaveModlFile = SaveModlPath + 'model' + string + '.h5'

    # load model
    model = load_model(SaveModlFile,custom_objects={'attention': attention})

    # test on test data
    predictions_test = model.predict(test_inp,batch_size=batchsize)

    # performance evaluation
    G_mean,f1_score,auc,mcc = Evaluate(test_oup, predictions_test)

    return G_mean,f1_score,auc,mcc


if __name__ == '__main__':
    # read the saved best weighted costs and hyperparameters
    ParaTable = pd.read_csv(config.ParaPath,dtype='string')

    G_mean_li = []
    f1_score_li = []
    auc_li = []
    mcc_li = []
    time_li = []
    for index,row in ParaTable.iterrows():
        round_no = int(row['round_no'])
        time = float(row['time'])
        Para = row['best_para']
        Para_r = Para.replace("[","").replace("]","")
        Para_s = Para_r.split(", ")
        Para_f = [float(item) for item in Para_s]
        print('Round: %s'%(round_no))
        G_mean,f1_score,auc,mcc = Test(Para_f, round_no)
        G_mean_li.append(G_mean)
        f1_score_li.append(f1_score)
        auc_li.append(auc)
        mcc_li.append(mcc)
        time_li.append(time)

    SaveDf = pd.DataFrame({'G_mean':G_mean_li,'F1_score':f1_score_li,'Auc':auc_li,'mcc':mcc_li,'Time':time_li})
    SaveDf.to_csv(config.SavePath,index=False)
    SaveDf = SaveDf.astype(float)

    mean_ = SaveDf.mean()
    std_ = SaveDf.std()

    print('#########################')
    print('mean values')
    print(mean_)
    print('#########################')
    print('standard deviation')
    print(std_)

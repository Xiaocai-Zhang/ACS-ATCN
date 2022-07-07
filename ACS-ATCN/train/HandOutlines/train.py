import pandas as pd
pd.options.mode.chained_assignment = None
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from format.format_train import FormatData
from evaluate.evaluation_val import Evaluate
import numpy as np
import random
import pathlib
os.chdir(os.path.dirname(__file__))



class config:
    '''
    configure parameters and paths
    '''
    n_classes = 2
    TrainValRatio = [0.8, 0.2]
    TrainInpPath = '../../data/proced/HandOutlines/train_input.csv'
    TrainOupPath = '../../data/proced/HandOutlines/train_target.csv'
    ParaSavePath = '../../para/HandOutlines/parameter.csv'
    ModelSavePath = '../../save_model/HandOutlines/'

    bounds_cost = [(0, 1),(1,20)]
    # batch, timestep, filter, layer, kernel, dropout, lr
    bounds_other = [(10, 300),(10, 2709),(1, 100),(1, 10),(1, 100),(0,1),(0.001, 0.1)]
    bounds_all = bounds_cost + bounds_other
    F_c = 0.7
    EarlyStopStep = 3
    maxiter = 10
    F_list = [(0.8,1),(0.6,0.8),(0.4,0.6),(0.2,0.4),(0,0.2),(0,0.2),(0,0.2),(0,0.2),(0,0.2),(0,0.2)]
    k_list = [50, 30, 20, 10, 5, 5, 5, 5, 5, 5]  # length of popsize must be equal to maxiter


class TempoConvNetworks:
    def TcnBlock(self, x, dilation_rate, nb_filters, kernel_size, dropout, padding):
        '''
        define TCN block
        '''
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']
        conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                       kernel_initializer=init)
        batch1 = BatchNormalization(axis=-1)
        ac1 = Activation('relu')
        drop1 = GaussianDropout(dropout)

        conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                       kernel_initializer=init)
        batch2 = BatchNormalization(axis=-1)
        ac2 = Activation('relu')
        drop2 = GaussianDropout(dropout)

        downsample = Conv1D(filters=nb_filters, kernel_size=1, padding='same', kernel_initializer=init)
        ac3 = Activation('relu')

        pre_x = x

        x = conv1(x)
        x = batch1(x)
        x = ac1(x)
        x = drop1(x)
        x = conv2(x)
        x = batch2(x)
        x = ac2(x)
        x = drop2(x)

        if pre_x.shape[-1] != x.shape[-1]:  # to match the dimensions
            pre_x = downsample(pre_x)

        assert pre_x.shape[-1] == x.shape[-1]

        try:
            out = ac3(pre_x + x)
        except:
            pre_x = tf.cast(pre_x,dtype=tf.float16)
            out = ac3(pre_x + x)

        return out

    def TcnNet(self, input, num_channels, kernel_size, dropout):
        '''
        define TCN network
        '''
        assert isinstance(num_channels, list)
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_rate = 2 ** i
            input = self.TcnBlock(input, dilation_rate, num_channels[i], kernel_size, dropout, padding='causal')

        out = input

        return out


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


def Train(x,round_no,f):
    '''
    train an ATCN model
    '''
    cost_1 = round(x[0], 4)
    ratio = round(x[1], 4)
    cost_2 = round(cost_1*ratio,4)
    batchsize = int(round(x[2]))
    timestep = int(round(x[3]))
    channel = int(round(x[4]))
    layer = int(round(x[5]))
    kernel_tcn = int(round(x[6]))
    dropout = round(x[7], 4)
    lr = round(x[8], 4)
    num_channels = [channel] * layer

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # mix precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    LastIdx = [idx for idx in range(2709)]
    ValdIdx = LastIdx[-timestep:]

    train_inp_loc = train_inp_glo[:, ValdIdx, :]
    val_inp_loc = val_inp_glo[:, ValdIdx, :]

    # build deep learning model
    inp_shape = (timestep, 1)
    input = Input(shape=inp_shape)

    # TCN block
    TCNetworks = TempoConvNetworks()
    output = TCNetworks.TcnNet(input, num_channels, kernel_tcn, dropout)

    # attention layer
    output = attention(return_sequences=False)(output)

    # desne layer
    output = Dense(config.n_classes)(output)
    output = Activation('softmax', dtype='float32')(output)

    model = Model(inputs=input, outputs=output)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    # shuffle training data
    np.random.seed(46)
    train_inp_loc = np.random.permutation(train_inp_loc)
    np.random.seed(46)
    train_oup_loc = np.random.permutation(train_oup_glo)

    string = ""
    for i in range(len(x)):
        if i in [2, 3, 4, 5, 6]:
            item = int(round(x[i]))
        else:
            item = round(x[i], 4)
        string = string + '_' + str(item)

    SaveModlPath = config.ModelSavePath+'round_'+str(round_no)+'/'
    SaveModlFile = SaveModlPath+'model' + string + '.h5'
    pathlib.Path(SaveModlPath).mkdir(parents=True, exist_ok=True)

    # define callbacks
    mcp_save = callbacks.ModelCheckpoint(SaveModlFile, save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10)

    class_weight = {0: cost_1,
                    1: cost_2}

    model.fit(x=train_inp_loc, y=train_oup_loc, epochs=100, class_weight=class_weight,
            batch_size=batchsize, validation_data=(val_inp_loc, val_oup_glo),
            callbacks=[mcp_save, early_stopping],
            verbose=0)

    # load model
    try:
        model = load_model(SaveModlFile,custom_objects={'attention': attention})
    except:
        pass

    # test on test data
    predictions_val = model.predict(val_inp_loc, batch_size=batchsize)

    # performance evaluation
    G_mean_val = Evaluate(val_oup_glo, predictions_val)

    K.clear_session()
    del model

    printtxt = "G_mean: %.4f: %s,%s,%s,%s,%s,%s,%s,%s,%s"%(G_mean_val,cost_1,ratio,batchsize,timestep,channel,layer,kernel_tcn,dropout,lr)
    print(printtxt)
    os.write(f, str.encode(printtxt+'\n'))

    return G_mean_val


def SelectTopK(InitialArray,BestGmean,step,k):
    '''
    function to select the top k candidates
    '''
    if step == 0:
        return InitialArray,BestGmean
    else:
        topkidx = sorted(range(len(BestGmean)), key=lambda i: BestGmean[i])[-k:]
        topkidx.sort()
        topkarray = [InitialArray[i] for i in topkidx]
        BestGmean_ = [BestGmean[i] for i in topkidx]

        return topkarray,BestGmean_


def RunDE(round_no):
    '''
    run ATDE algorithm
    '''
    start = datetime.now()
    from de.differential_evolution import DEAlgorithm

    DEAlgorithmClass = DEAlgorithm(config)
    initial_popusize = config.k_list[0]
    InitialArray = DEAlgorithmClass.initialization(initial_popusize)

    GlobalMaxValLi = []
    BestGmean = []
    GlobalMaxVal = 0

    SaveTextPath = '../../text/HandOutlines/round_' + str(round_no) + '/'
    pathlib.Path(SaveTextPath).mkdir(parents=True, exist_ok=True)
    f = os.open(SaveTextPath+"prcoess.txt", os.O_RDWR | os.O_CREAT)

    for step in range(config.maxiter):
        printtxt = 'Round: %s, Step %s' % (round_no,step)
        print(printtxt)
        os.write(f, str.encode(printtxt+"\n"))

        # select top-k candidates
        k = config.k_list[step]
        InitialArray,BestGmean = SelectTopK(InitialArray, BestGmean, step, k)

        F_l = config.F_list[step][0]
        F_u = config.F_list[step][1]
        MutatedArray = DEAlgorithmClass.mutation(InitialArray,F_l,F_u)
        CrossArray = DEAlgorithmClass.crossover(InitialArray, MutatedArray)

        if step == 0:
            args = InitialArray + CrossArray
            CombineArray = args
        else:
            args = CrossArray
            CombineArray = InitialArray + CrossArray

        gmeanlist = []
        for num in range(len(args)):
            item = args[num]
            try:
                res = Train(item,round_no,f)
            except:
                res = 0
            gmeanlist.append(res)

        if step == 0:
            gmeanList = gmeanlist
        else:
            gmeanList = BestGmean + gmeanlist

        StepMaxVal = max(gmeanList)

        if StepMaxVal > GlobalMaxVal:
            LIdx = gmeanList.index(StepMaxVal)
            StepOptimalPara = CombineArray[LIdx]
            GlobalOptimalPara = StepOptimalPara
            GlobalMaxVal = StepMaxVal

        printtxt = "Step: %s, max G_mean: %s, hyperparameter: %s" % (step, GlobalMaxVal, GlobalOptimalPara)
        print(printtxt)
        os.write(f, str.encode(printtxt+'\n'))

        assert len(InitialArray) == len(CrossArray) == k
        SelectArray, BestGmean = DEAlgorithmClass.selection(gmeanList, InitialArray, CrossArray, k)
        InitialArray = SelectArray

        # early stopping applied
        GlobalMaxValLi.append(GlobalMaxVal)
        if step + 1 >= config.EarlyStopStep:
            if GlobalMaxValLi[-1] == GlobalMaxValLi[-2] == GlobalMaxValLi[-3]:
                break

    duration = (datetime.now() - start).total_seconds()/3600
    printtxt = "computational time: %s h" %(duration)
    print(printtxt)
    os.write(f, str.encode(printtxt + '\n'))

    # update best para table
    Df_Para = pd.read_csv(config.ParaSavePath)
    Df_step = pd.DataFrame({'round_no':[round_no],'best_para':[GlobalOptimalPara],'time':[duration],'best_fitness':[GlobalMaxVal]})
    Df_Para = pd.concat([Df_Para,Df_step],axis=0)
    Df_Para.to_csv(config.ParaSavePath, index=False)

    return None


if __name__ == '__main__':
    # read data
    train_inp = pd.read_csv(config.TrainInpPath, header=None, dtype=np.float16)
    train_oup = pd.read_csv(config.TrainOupPath, header=None, dtype=np.float16)

    # process data, raw data alread normalized
    FDClass = FormatData(train_inp, train_oup)
    train_inp, train_oup = FDClass.main()

    # split validation data
    num_val = round(train_inp.shape[0] * config.TrainValRatio[1])
    IdxTrainVal = [idx for idx in range(0, train_inp.shape[0])]

    # get indices for val set
    random.seed(191)
    IdxVal = random.sample(IdxTrainVal, num_val)

    # define global variables
    global train_inp_glo,train_oup_glo,val_inp_glo,val_oup_glo

    # get indices for train set
    IdxTrain = [idx for idx in IdxTrainVal if idx not in IdxVal]
    val_inp_glo = train_inp[IdxVal, :, :]
    val_oup_glo = train_oup[IdxVal, :]
    train_inp_glo = train_inp[IdxTrain, :, :]
    train_oup_glo = train_oup[IdxTrain, :]

    # run 10 times
    round_li = list(range(1,10))
    [RunDE(num) for num in round_li]

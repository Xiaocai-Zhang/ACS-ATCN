# ACS-ATCN
## Setup
Code was developed and tested on Ubuntu 18.04 with Python 3.8 and TensorFlow 2.5.0. You can setup a virtual environment by running the code like this:
```
virtualenv env --python=python3.8
source env/bin/activate
cd ACS-ATCN
pip3 install -r requirements.txt
```
## Download Data Sets
Run the following commands to download data sets from cloud.
```
gdown https://drive.google.com/uc?id=1GWhAaP3EgCNd0p_mwdDBk7KL9AQO82hm
unzip data
rm data.zip
```
## Test Models
Firstly, run the following codes to download the parameters.
```
gdown https://drive.google.com/uc?id=1FIU6lYU_516dQobQBmrWmq87n76dgOWy
unzip para
rm para.zip
```
Then, you can run the following command to download the trained models.
```
gdown https://drive.google.com/uc?id=1mdtcPJayQ-RVEShrw5WLKNXEyomKZnA9
unzip save_model
rm save_model.zip
```
Finally, you can run the following command to replicate the results.
```
python3 test/XXXX/test.py
```
For example, for the FordA data set, run
```
python3 test/FordA/test.py
```
The results are saved under the "./para/" folder.
## Train Models
You can run the following command to train your own models.
```
python3 train/XXXX/train.py
```
For example, for training the models on FordA data set, run command
```
python3 train/FordA/train.py
```

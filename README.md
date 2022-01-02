# ACS-ATCN
## Setup
Code was developed and tested on Ubuntu 18.04 with Python 3.8 and TensorFlow 2.5.0. You can setup a virtual environment by running the code like this:
```
python3 -m venv env
source env/bin/activate
cd ACS-ATCN
pip3 install -r requirements.txt
```
## Download Data Sets
Run the following commands to download data sets from cloud.
```
gdown https://drive.google.com/uc?id=1GWhAaP3EgCNd0p_mwdDBk7KL9AQO82hm
unzip data
```
## Test Models
Firstly, run the following codes to download the parameters.
```
gdown https://drive.google.com/uc?id=1FIU6lYU_516dQobQBmrWmq87n76dgOWy
unzip para
```
Then, you can run the following command to download the trained models.
```
gdown https://drive.google.com/uc?id=1mdtcPJayQ-RVEShrw5WLKNXEyomKZnA9
unzip save_model
```

```
python3 test/XXXX/test_XXm.py
```
For example, for the I280-S 15-min prediction task, run
```
python3 test/I280-S/test_15m.py
```
The results are saved under the "./hypara/" folder.

# Incomplete Utterance Rewriting <img src="https://pytorch.org/assets/images/logo-dark.svg" height = "25" align=center />



The official pytorch implementation of our paper [How Well Apply Simple MLP to Incomplete Utterance Rewriting?]



## Content

- [Install Dependencies](#requirement)
- [Download and Preprocess Dataset](#data)
- [Train Model](#train)
- [Evaluate Model](#evaluate)


## Requirement

### Python Environment

First of all, you should setup a python environment. This code base has been tested under python 3.x, and we officially support python 3.7.

After installing python 3.7, we strongly recommend you to use `virtualenv` (a tool to create isolated Python environments) to manage the python environment. You could use following commands to create a environment.

```bash
python -m pip install virtualenv
virtualenv venv
```

### Activate Virtual Environment
Then you should activate the environment to install the dependencies. You could achieve it via using the command as below. (Please change $ENV_FOLDER to your own virtualenv folder path, e.g. venv)

```bash
$ENV_FOLDER\Scripts\activate.bat (Windows)
source $ENV_FOLDER/bin/activate (Linux)
```

### Install Libraries

The most important requirements of our code base are as following:
- pytorch >= 1.2.0 (not tested on other versions, but 1.0.0 may work though)
- allennlp == 0.9.0

Other dependencies can be installed by

```console
pip install -r requirement.txt
```

## Data

### Prepare Dataset

Although we cannot provide dataset resources (copyright issue) in our repo, we provide `download.sh` for automatically downloading and preprocessing datasets used in our paper.





## Train

You could train models on different datasets using `*.sh` files under the `src` folder.  For example, you could train `MIUR` on `Restoration-200K (multi)` by running the following command under the `src` folder as:

```console
./train_multi.sh
```




## Evaluate

Once a model is well trained, `allennlp` will save a compressed model zip file which is usually named after `model.tar.gz` under the checkpoint folder. Our evaluation is based on it. We provide a evaluate file under `src` folder, and you could evaluate a model file by running the following command:

```concolse
python evaluate.py --model_file model.tar.gz --test_file ../dataset/Multi/test.txt
```
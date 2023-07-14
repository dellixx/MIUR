<h2 align="center">
 How Well Apply Simple MLP to Incomplete Utterance Rewriting?  <img src="https://pytorch.org/assets/images/logo-dark.svg" height = "20" align=center />
</h2>

<p align="center">
  <img src="https://img.shields.io/badge/ACL-2023-brightgreen">
  <a href = 'https://aclanthology.org/2023.acl-short.134.pdf' target='_blank'><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
</p>


<p align="center">
Codes for the paper How Well Apply Simple MLP to Incomplete Utterance Rewriting?  accepted by the ACL 2023.
</p>




## Content

- [Install Dependencies](#requirement)
- [Download and Preprocess Dataset](#data)
- [Train Model](#train)
- [Evaluate Model](#evaluate)
- [Pre-trained Models](#pre-trained-models)


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
python evaluate.py --model_file ../checkpoints/multi/model.tar.gz --test_file ../dataset/Multi/test.txt
```

The above script will generate a file `model.tar.gz.json` which records the detailed performance. For example, the performance of `MIUR` on `Restoration-200K` is:
```json
{
    "ROUGE": 0.895832385848569,
    "_ROUGE1": 0.9262545735851855,
    "_ROUGE2": 0.8578286223419522,
    "EM": 0.510384012539185,
    "_P1": 0.7645602605863192,
    "_R1": 0.6377567655689599,
    "F1": 0.6954254562692581,
    "_P2": 0.6270252754374595,
    "_R2": 0.5279672578444747,
    "F2": 0.5732484076433121,
    "_P3": 0.543046357615894,
    "_R3": 0.4591725867112411,
    "F3": 0.49759985508559007,
    "_BLEU1": 0.9300601956358164,
    "_BLEU2": 0.9015189890585196,
    "_BLEU3": 0.8741648040269356,
    "BLEU4": 0.8467568893283197,
    "loss": 0.018303699255265087
}
```
Next, we will provide all pre-trained models to reproduce results reported in our paper. We recommend you to download them and put them into the folder pretrained_weights and run commands like below:

```concolse
python evaluate.py --model_file ../pretrianed_weights/multi.tar.gz --test_file ../dataset/Multi/test.txt
```

## Pre-trained Models


| Dataset | Config | Pretrained_Weights |
| :---: | :---: | :---: |
| Multi (Restoration-200K) | multi.jsonnet | [multi.tar.gz](https://drive.google.com/file/d/1uRrbpqOw1Nga1maSnX0gWF1kSp0ncB48/view?usp=share_link) |
| Rewrite | rewrite.jsonnet | [rewrite.tar.gz](https://drive.google.com/file/d/1zxUzAeZcktGprjl2mcg00GPx-jEGWpN-/view?usp=share_link)|
| CANARD | canard.jsonnet | [canard.tar.gz](https://drive.google.com/file/d/14ZDIUkZi8UqoIvtv6lJJMJvQQ_eJrYPp/view?usp=share_link) |



#### Acknowledgement

We refer to the code of [RUN](https://github.com/microsoft/ContextualSP/tree/master/incomplete_utterance_rewriting). Thanks for their contributions.

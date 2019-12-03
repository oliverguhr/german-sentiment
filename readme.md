# Broad-Coverage German Sentiment Classification Model for Dialog Systems

his repository contains the code and data for the Paper "Training a Broad-Coverage German Sentiment Classification Model for Dialog
Systems"

## Data Sets

The data sets can be found in the [source-data](source-data/) folder. Due to legal requirements, we can not provide the SCARE data in this repository, but you can [obtain the data from the author directly](http://www.romanklinger.de/scare/).

The data sets that we used to train our models are located within the [training-data](training-data/) folder. Since we can not redistribute the Scare data set publicly, we can not provide the data sets that contain fragments of this data. However, if you are interested in this data, please write a mail to oliver.guhr-at-htw-dresden.de.

## Trained Models

You can find the trained models for FastText and Bert in the [models](./models) folder. We provide these trained models:


| Note                        |   FastText    |    Bert       |
|:----------------------------|--------------:|--------------:|
| trained on balanced data    |  [model](models/paper-balanced/fasttext)             |  [model](models/paper-balanced/bert) |
| trained on unbalanced data  |        [model](models/paper-unbalanced/fasttext) |        [model](models/paper-unbalanced/bert) |
| trained with out scare      |       [model](models/paper-no-scare-balanced/fasttext)|       [model](models/paper-no-scare-balanced/bert) |


## Setup

We recommend to install this project in a python virtual environment. To install and activate this virtual environment you need to execute this three commands. 

```bash
pip3 install virtualenv
virtualenv venv
source venv/bin/activate
```
Make sure that you are using a recent python version by running "python -V ". You should at least run Python 3.6.

```bash
python -V 
> Python 3.6.8
```

Next, install the needed python packages.

```bash
pip install -r requirements.txt
```

If you want to use the bert model, you need to download the pretrained german bert model.

```bash

mkdir  bert/bert-base-german-cased && cd bert/bert-base-german-cased

wget https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-pytorch_model.bin -o pytorch_model.bin

wget https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-config.json -o config.json

wget https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt -o vocab.txt
```
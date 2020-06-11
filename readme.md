# Broad-Coverage German Sentiment Classification Model for Dialog Systems

This repository contains the code and data for the Paper ["Training a Broad-Coverage German Sentiment Classification Model for Dialog Systems"](https://www.aclweb.org/anthology/2020.lrec-1.202.pdf) published at LREC 2020.

## Usage

If you like to use the models for your own projects please head over to [this repository.](https://github.com/oliverguhr/german-sentiment-lib) It contains a Python package that provides a easy to use interface.

## Data Sets

The data sets can be downloaded from [here](https://zenodo.org/record/3693810/). Due to legal requirements, we can not provide the SCARE data set in this repository, but you can [obtain the data from the author directly](http://www.romanklinger.de/scare/).

The preprocessed data sets that we used to train our models can be downloaded from [here (600 MB zip)](https://zenodo.org/record/3693810/files/no-scare-balanced.zip?download=1). This file does not contain all the training files, since we can not redistribute the *Scare* data set publicly. However, if you are interested in this data, please write a mail to oliver.guhr-at-htw-dresden.de.

The unprocessed data set can be downloaded from [here (1.5 GB)](https://zenodo.org/record/3693810/files/sentiment-data-reviews-and-neutral.zip?download=1), it contains all hotel and movie reviews, plus a set of neutral german texts.


## Trained Models

You can download our trained models for FastText and Bert [here (6 GB)](https://zenodo.org/record/3693810/files/models.zip?download=1).

## Setup

We recommend to install this project in a python virtual environment. To install and activate this virtual environment you need to execute this three commands. 

```bash
pip3 install virtualenv
python3 -m venv ./venv
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

In order to reproduce the results, you need to download our models and data. We provide a script that downloads all required packages:

```bash
sh download-models-and-data.sh
```


## Cite

You can read the [paper here](https://www.aclweb.org/anthology/2020.lrec-1.202.pdf). Please cite us if you found this useful. 

```
@InProceedings{guhr-EtAl:2020:LREC,
  author    = {Guhr, Oliver  and  Schumann, Anne-Kathrin  and  Bahrmann, Frank  and  Böhme, Hans Joachim},
  title     = {Training a Broad-Coverage German Sentiment Classification Model for Dialog Systems},
  booktitle      = {Proceedings of The 12th Language Resources and Evaluation Conference},
  month          = {May},
  year           = {2020},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {1620--1625},
  url       = {https://www.aclweb.org/anthology/2020.lrec-1.202/}
}
```

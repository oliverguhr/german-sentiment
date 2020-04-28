# Broad-Coverage German Sentiment Classification Model for Dialog Systems

his repository contains the code and data for the Paper "Training a Broad-Coverage German Sentiment Classification Model for Dialog
Systems"

## Data Sets

The data sets can be found in the [source-data](source-data/) folder. Due to legal requirements, we can not provide the SCARE data in this repository, but you can [obtain the data from the author directly](http://www.romanklinger.de/scare/).

The preprocessed data sets that we used to train our models can be obtained from [here (600 MB zip)](https://www2.htw-dresden.de/~guhr/dist/sentiment/no-scare-balanced.zip). This file does not contain all the training files, since we can not redistribute the *Scare* data set publicly. However, if you are interested in this data, please write a mail to oliver.guhr-at-htw-dresden.de.

The unprocessed data set can be obtained from [here (1.5 GB)](https://www2.htw-dresden.de/~guhr/dist/sentiment/no-scare-balanced.zip), it contains all hotel and movie reviews, plus a set of neutral german texts.


## Trained Models

You can find our trained models for FastText and Bert in the [(here 5 GB)](https://www2.htw-dresden.de/~guhr/dist/sentiment/models.zip) folder. 



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

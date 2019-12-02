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

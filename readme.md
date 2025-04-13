# Broad-Coverage German Sentiment Classification Model for Dialog Systems

This repository contains the code and data for the Paper ["Training a Broad-Coverage German Sentiment Classification Model for Dialog Systems"](https://www.aclweb.org/anthology/2020.lrec-1.202.pdf) published at LREC 2020.

## Usage

If you like to use the models for your own projects please head over to [this repository.](https://github.com/oliverguhr/german-sentiment-lib) It contains a Python package that provides a easy to use interface.

## Data Sets

We trained our models on a combination of self created and exisiting data sets, to cover a broad variety of topics and domains.

| **Data Set**      | **Positive Samples** | **Neutral Samples** | **Negative Samples** | **Total Samples** |
| :---------------- | -------------------: | ------------------: | -------------------: | ----------------: |
| [Emotions](https://github.com/oliverguhr/german-sentiment)          |                  188 |                  28 |                1,090 |             1,306 |
| [filmstarts](https://github.com/oliverguhr/german-sentiment)        |               40,049 |                   0 |               15,610 |            55,659 |
| [GermEval-2017](https://sites.google.com/view/germeval2017-absa/home)     |                1,371 |              16,309 |                5,845 |            23,525 |
| [holidaycheck](https://github.com/oliverguhr/german-sentiment)      |            3,135,449 |                   0 |              388,744 |         3,524,193 |
| [Leipzig Wikipedia Corpus 2016](https://wortschatz.uni-leipzig.de/de/download/german) |                    0 |           1,000,000 |                    0 |         1,000,000 |
| [PotTS](https://www.aclweb.org/anthology/L16-1181/)              |                3,448 |               2,487 |                1,569 |             7,504 |
| [SB10k](https://www.spinningbytes.com/resources/germansentiment/)             |                1,716 |               4,628 |                1,130 |             7,474 |
| [SCARE](https://www.romanklinger.de/scare/)             |              538,103 |                   0 |              197,279 |           735,382 |
| **Sum**           |        **3,720,324** |       **1,023,452** |          **611,267** |     **5,355,043** |

The data sets without the SCARE Dataset can be downloaded from [here](https://zenodo.org/record/3693810/). Due to legal requirements, we can not provide the SCARE data set directly, but you can [obtain the data from the author directly](http://www.romanklinger.de/scare/). However, if you are interested in this data, please obtain the Scare data set from the autors and integrate it usign our provided scripts to create the combined data set.

The unprocessed data set can be downloaded from [here (1.5 GB)](https://zenodo.org/record/3693810/files/sentiment-data-reviews-and-neutral.zip?download=1), it contains all hotel and movie reviews, plus a set of neutral german texts.

The **Filmstarts** data set consists of 71,229 user written movie
reviews in the German language. We have collected this data from the
German website filmstarts.de using a web crawler. The users can label
their reviews in the range of 0.5 to 5 stars. With 40,049 documents the
majority of the reviews in this data set are positive and only 15,610
reviews are negative. All data was downloaded between the 15th and 16th
of October 2018, containing reviews up to this date.

The **holidaycheck** data set contains hotel reviews from the German
website holidaycheck.de. The users of this website can write a general
review and rate their hotel. Additionally, they can review and rate six
specific aspects: location & surroundings, rooms, service, cuisine,
sports & entertainment and hotel. A full review contains therefore seven
texts and the associated star rating in the range from zero to six
stars. In total, we have downloaded 4,832,001 text-rating pairs for
hotels from ten destinations: Egypt, Bulgaria, China, Greece, India,
Majorca, Mexico, Tenerife, Thailand and Tunisia. The reviews were
obtained from November to December 2018 and contain reviews up to this
date. After removing all reviews with no stars or four stars, the data
set contains 3,524,193 text-rating pairs.

The **Emotions** data set contains a list of utterances that we have
recorded during the "Wizard of Oz" experiments with the service robots.
We have noticed, that people used insults while talking to the robot.
Since most of these words are filtered in social media and review
platforms, other data sets do not contain such words. We used synonym
replacement as a data augmentation technique to generate new utterances
based on our recordings. Besides negative feedback, this data set
contains also positive feedback and phrases about sexual identity and
orientation that where labelled as neutral. Overall this data set
contains 1,306 examples.

## Trained Models

You can download our trained models for FastText and Bert [here (6 GB)](https://zenodo.org/record/3693810/files/models.zip?download=1). With this models we achived following results:

### Bert

| **Data Set**      | **Balanced** | **Unbalanced** |
| :---------------- | -----------: | -------------: |
| [SCARE](https://www.romanklinger.de/scare/)                 |       0.9409 |     **0.9436** |
| [GermEval-2017](https://sites.google.com/view/germeval2017-absa/home)     |       0.7727 |     **0.7885** |
| [holidaycheck](https://github.com/oliverguhr/german-sentiment)      |       0.9552 |     **0.9775** |
| [SB10k](https://www.spinningbytes.com/resources/germansentiment/)             |   **0.6930** |         0.6720 |
| [filmstarts](https://github.com/oliverguhr/german-sentiment)        |       0.9062 |     **0.9219** |
| [PotTS](https://www.aclweb.org/anthology/L16-1181/)              |       0.6423 |     **0.6502** |
| [emotions](https://github.com/oliverguhr/german-sentiment)          |   **0.9652** |         0.9621 |
| [Leipzig Wikipedia Corpus 2016](https://wortschatz.uni-leipzig.de/de/download/german) |   **0.9983** |         0.9981 |
| combined          |       0.9636 |     **0.9744** |

Micro averaged F1 scores for BERT trained on the balanced and unbalanced data set.

### Fast Text

| **Data Set**      | **Balanced** | **Unbalanced** |
| :---------------- | -----------: | -------------: |
| [SCARE](https://www.romanklinger.de/scare/)                 |       0.9071 |     **0.9083** |
| [GermEval-2017](https://sites.google.com/view/germeval2017-absa/home)     |       0.6970 |     **0.6980** |
| [holidaycheck](https://github.com/oliverguhr/german-sentiment)      |       0.9296 |     **0.9639** |
| [SB10k](https://www.spinningbytes.com/resources/germansentiment/)             |   **0.6862** |         0.6213 |
| [filmstarts](https://github.com/oliverguhr/german-sentiment)        |       0.8206 |     **0.8432** |
| [PotTS](https://www.aclweb.org/anthology/L16-1181/)              |       0.5268 |     **0.5416** |
| [emotions](https://github.com/oliverguhr/german-sentiment)          |   **0.9913** |         0.9773 |
| [Leipzig Wikipedia Corpus 2016](https://wortschatz.uni-leipzig.de/de/download/german) |       0.9883 |     **0.9886** |
| combined          |       0.9405 |     **0.9573** |

Micro averaged F1 scores for FastText trained on the balanced and
unbalanced.

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
> Python 3.11.0
```

Next, install the needed python packages.

```bash
pip install -r requirements-3.11.txt
```

In order to reproduce the results, you need to download our models and data. We provide a script that downloads all required packages:

```bash
sh download-models-and-data.sh
```

## Paper & Citetation

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

If you use the combined data set for your work, you can use this list to cite all the contained data sets:

```
@LanguageResource{sanger_scare_2016,
	address = {Portorož, Slovenia},
	title = {{SCARE} ― {The} {Sentiment} {Corpus} of {App} {Reviews} with {Fine}-grained {Annotations} in {German}},
	url = {https://www.aclweb.org/anthology/L16-1178},	
	urldate = {2019-11-07},
	booktitle = {Proceedings of the {Tenth} {International} {Conference} on {Language} {Resources} and {Evaluation} ({LREC}'16)},
	publisher = {European Language Resources Association (ELRA)},
	author = {Sänger, Mario and Leser, Ulf and Kemmerer, Steffen and Adolphs, Peter and Klinger, Roman},
	year = {2016},
	pages = {1114--1121}
}

@LanguageResource{sidarenka_potts:_2016,
	address = {Paris, France},
	title = {{PotTS}: {The} {Potsdam} {Twitter} {Sentiment} {Corpus}},
	isbn = {978-2-9517408-9-1},
	language = {english},
	booktitle = {Proceedings of the {Tenth} {International} {Conference} on {Language} {Resources} and {Evaluation} ({LREC} 2016)},
	publisher = {European Language Resources Association (ELRA)},
	author = {Sidarenka, Uladzimir},
	editor = {Chair), Nicoletta Calzolari (Conference and Choukri, Khalid and Declerck, Thierry and Goggi, Sara and Grobelnik, Marko and Maegaard, Bente and Mariani, Joseph and Mazo, Helene and Moreno, Asuncion and Odijk, Jan and Piperidis, Stelios},
	year = {2016},
	note = {event-place: Portorož, Slovenia}
}

@LanguageResource{cieliebak_twitter_2017,
	address = {Valencia, Spain},
	title = {A {Twitter} {Corpus} and {Benchmark} {Resources} for {German} {Sentiment} {Analysis}},
	url = {https://www.aclweb.org/anthology/W17-1106},
	doi = {10.18653/v1/W17-1106},
	urldate = {2019-11-07},
	booktitle = {Proceedings of the {Fifth} {International} {Workshop} on {Natural} {Language} {Processing} for {Social} {Media}},
	publisher = {Association for Computational Linguistics},
	author = {Cieliebak, Mark and Deriu, Jan Milan and Egger, Dominic and Uzdilli, Fatih},
	month = apr,
	year = {2017},
	pages = {45--51}
}

@LanguageResource{wojatzki_germeval_2017,
	address = {Berlin, Germany},
	title = {{GermEval} 2017: {Shared} {Task} on {Aspect}-based {Sentiment} in {Social} {Media} {Customer} {Feedback}},
	booktitle = {Proceedings of the {GermEval} 2017 – {Shared} {Task} on {Aspect}-based {Sentiment} in {Social} {Media} {Customer} {Feedback}},
	author = {Wojatzki, Michael and Ruppert, Eugen and Holschneider, Sarah and Zesch, Torsten and Biemann, Chris},
	year = {2017},
	pages = {1--12}	
}

@inproceedings{goldhahn-etal-2012-building,
    title = "Building Large Monolingual Dictionaries at the Leipzig Corpora Collection: From 100 to 200 Languages",
    author = "Goldhahn, Dirk  and
      Eckart, Thomas  and
      Quasthoff, Uwe",
    booktitle = "Proceedings of the Eighth International Conference on Language Resources and Evaluation ({LREC}'12)",
    month = may,
    year = "2012",
    address = "Istanbul, Turkey",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2012/pdf/327_Paper.pdf",
    pages = "759--765"
}
```

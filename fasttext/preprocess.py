import glob
import os
import os.path
import random
import subprocess
from tabulate import tabulate
import textcleaner
import tools

random.seed(42)

source_data ="../source-data/"
path = "./modeldata/"
pathForSets = path+"sets/"
if not os.path.exists(pathForSets):
    os.makedirs(pathForSets)

pathForTestsets = path+"test/"
if not os.path.exists(pathForTestsets):
    os.makedirs(pathForTestsets)

def fileNameFor(name, setName="", classname="", folder=pathForSets):
    filename = name
    if(setName is not ""):
        filename += "."+setName
    if(classname is not ""):
        filename += "."+classname
    return folder+filename


def save(name, className, data):
    trainValidPath = fileNameFor(name, "", className)
    with open(trainValidPath, 'w') as f:
        f.writelines(["{}\t__label__{}\t{}\n".format(name,line[0], line[1]) for line in data])


def saveAndSplit(name, className, data):
    # old function, remove it
    test, trainValid = tools.splitByRatio(data, 0.1)

    trainValidPath = fileNameFor(name, "", className)
    with open(trainValidPath, 'w') as f:
        f.writelines(["__label__{} {}\n".format(line[0], line[1])
                      for line in trainValid])

    testPath = fileNameFor(name, "test", "", pathForTestsets)
    with open(testPath, 'a') as f:
        f.writelines(["__label__{} {}\n".format(line[0], line[1])
                      for line in test])


def cleanAndSplit(name, dataLoader):

    isSetAlreadyLoaded = os.path.exists(fileNameFor(name, "", "neutral")) or os.path.exists(
        fileNameFor(name, "", "negative")) or os.path.exists(fileNameFor(name, "", "positive"))

    if isSetAlreadyLoaded == False:
        print("cleaning set: "+name)
        data = textcleaner.cleanData(dataLoader())        
        neutral, negativ, positiv = tools.splitPerClass(data)

        save(name, "neutral", neutral)
        save(name, "negative", negativ)
        save(name, "positive", positiv)
    return [isSetAlreadyLoaded, len(positiv), len(neutral), len(negativ), len(positiv) + len(neutral) + len(negativ)]


def executeToFile(command, filePath, mode="w", shellMode=False):
    with open(filePath, mode) as outfile:
        subprocess.call(command, stdout=outfile, shell=shellMode)

def createSetForClass(classname, dataSets):
    """concats all datasets to single datatset file for a given class"""
    command = ["cat"] + [fileNameFor(dataSet, "", classname)
                         for dataSet in dataSets]

    filePath = fileNameFor("all", "", classname)

    executeToFile(command, filePath)
    
    return tools.lineCount(filePath)

def splitTrainValidTest(data, train, valid, test):    
    trainCount = round(len(data) * train)
    validCount = round(len(data) * valid)
    train = data[:trainCount]
    valid = data[trainCount:trainCount+validCount]
    test = data[trainCount+validCount:]
    return train, valid, test

def split(count, className):
    """creates a train / valid split"""
    # could be done via commandline:
    #f"shuf {fileName} | tee >(head -n {trainCount} > train.file) >(tail -n {testCount} > test.file) "

    fileName = fileNameFor("all", "", className)
    data = tools.readAllLines(fileName)
    random.shuffle(data)
    
    data = data[:count]
    
    train, valid, test = splitTrainValidTest(data, .7, .2, .1)

    tools.writeAllLines(fileNameFor("all", "train", className), train)
    tools.writeAllLines(fileNameFor("all", "valid", className), valid)
    tools.writeAllLines(fileNameFor("all", "test", className), test)


def run():
    if(tools.config()["preprocessing"]["use-cache"] is False):
        subprocess.call(
            f"rm -rf {pathForSets}* {pathForTestsets}* ", shell=True)

    dataLoaders = [
        ["emotions", lambda:tools.loadData(source_data+"emotions")],
        ["germeval", lambda:tools.loadGermeval2017(source_data+"germeval2017/set_v1.4.tsv")],
        ["sb10k", lambda:tools.loadData(source_data+"SB10k/not-preprocessed/corpus_label_text.tsv","\t")],
        ["PotTS", lambda:tools.loadData(source_data+"PotTS/not-preprocessed/corpus_label_text.tsv","\t")],
        ["filmstarts", lambda:tools.loadFilmstarts(source_data+"filmstarts/filmstarts.tsv")],
        ["scare", lambda:tools.loadScareSet(source_data+"scare_v1.0.0_data/reviews/")],
        ["holidaycheck", lambda:tools.loadHolidaycheck(source_data+"holidaycheck/holidaycheck.clean.filtered.tsv")],
        ["leipzig-mixed-typical-2011", lambda:tools.loadData(source_data+"leipzig/deu-mixed-labeled")],
        ["leipzig-newscrawl-2017", lambda:tools.loadData(source_data+"leipzig/deu-newscrawl-2017-labeled")],
        ["leipzig-deu-wikipedia-2016", lambda:tools.loadData(source_data+"leipzig/deu-wikipedia-2016-labeled")]
    ]
    dataSets = []
    table = []
    dataSetsToLoad = tools.config()["datasets"]

    for dataSet in dataSetsToLoad:
        if dataSet["train"] is True or dataSet["test"] is True:
            # if this fails the loader you are defined in the config, is not defined in the code
            loader = next(
                filter(lambda x: x[0] == dataSet["name"], dataLoaders))

            # split every set in its 3 classes
            meta_info = cleanAndSplit(*loader)

            if dataSet["train"] is True:
                dataSets.append(loader)

            table.append(list(dataSet.values())+meta_info)

    headers = ["set name", "training", "test", "from cache", "positiv","neutral","negative", "total"]
    print(tabulate(table, headers, tablefmt="pipe", floatfmt=".4f"))

    trainSets = [dataset["name"]
                 for dataset in dataSetsToLoad if dataset["train"] is True]

    # combine single datasets into one set per class
    neutralSamples = createSetForClass("neutral", trainSets)
    positiveSamples = createSetForClass("positive", trainSets)
    negativeSamples = createSetForClass("negative", trainSets)

    print("\nclass distribution in data set:")
    print("neutral \t{}\npostitive\t{}\nnegative\t{}".format(
        neutralSamples, positiveSamples, negativeSamples))

    # balance classes    
    if(tools.config()['preprocessing']['balance'] == 'down'):
       print("\nbalance classes with downsampling")
       samplesPerClass = min(neutralSamples, positiveSamples, negativeSamples)
       print("random sampels per class: {}".format(samplesPerClass))
       print("total sampels: {}".format(samplesPerClass * 3))
       # train / test split per class
       split(samplesPerClass, "neutral")
       split(samplesPerClass, "positive")
       split(samplesPerClass, "negative")
    else:           
       split(neutralSamples, "neutral")
       split(positiveSamples, "positive")
       split(negativeSamples, "negative") 
       print(f"random sampels per class neutral: {neutralSamples}")
       print(f"random sampels per class positiv: {positiveSamples}")
       print(f"random sampels per class negative:{negativeSamples}")
       print(f"total sampels: {neutralSamples +positiveSamples + negativeSamples}")

    # combine classes to set
    trainFile = path+"model.train"
    validFile = path+"model.valid"
    testFile = path+"model.test"
    
    executeToFile(f"cat {pathForSets}all.train.* | cut -f2,3", trainFile, shellMode=True)
    executeToFile(f"cat {pathForSets}all.valid.* | cut -f2,3", validFile, shellMode=True)
    executeToFile(f"cat {pathForSets}all.test.*", testFile, shellMode=True)
    
    totalTrain = tools.lineCount(trainFile)
    totalValid = tools.lineCount(validFile)
    totalTest = tools.lineCount(testFile)
    totalLines = float(totalTrain + totalValid + totalTest)

    print("\nsamples in:\ntrain\t{}\nvalid\t{}\ntest\t{}\nsum\t{}".format(
        totalTrain, totalValid, totalTest, totalLines))

    print("\npercentage in:\ntrain\t{}\nvalid\t{}\ntest\t{}".format(
        totalTrain/totalLines, totalValid/totalLines, totalTest/totalLines))

    test_sets = [dataset["name"]
                 for dataset in dataSetsToLoad if dataset["train"] is False and dataset["test"] is True]
    print(f"datasets just for testing {test_sets}")
    if os.path.exists(testFile+".extra"): os.remove(testFile+".extra")

    for test_set in test_sets:
       executeToFile(f"cat {pathForSets}{test_set}.* ", testFile+".extra",mode="a", shellMode=True) 

    executeToFile(f"cat {testFile} {testFile}.extra", path+"model.test.full", shellMode=True)

    executeToFile(f"cat {pathForSets}all.negative {pathForSets}all.neutral {pathForSets}all.positive | cut -f3 ", path+"wordvecc.train", shellMode=True)     
    
    # cleanup
    #subprocess.call("rm "+pathForSets+"all*",shell=True)

    # create wordvector training file
    #executeToFile(f"cat {pathForSets}*.*",path+"fasttext.wordvecc",shellMode = True)


if __name__ == '__main__':
    run()

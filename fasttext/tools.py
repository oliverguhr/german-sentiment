import glob
import json

config_field = None
def config():
    global config_field
    if(config_field is None):
        config_field = json.load(open('config.json'))
    return config_field

def splitPerClass(data):
    neutral = [line for line in data if line[0] == 'neutral']
    negativ = [line for line in data if line[0] == 'negative']
    positiv = [line for line in data if line[0] == 'positive']

    return neutral, negativ, positiv

def flatmap(list_of_lists):
    return [y for x in list_of_lists for y in x]

def lineCount(fname:str):    
    with open(fname) as f:   
        i = 0     
        for i, _ in enumerate(f, 1):
            pass
    return i

def splitByRatio(list:list,ratio:float):
    count = int(len(list)*ratio)
    return list[:count], list[count:]

def writeAllLines(path, lines):
    with open(path, 'w') as f:
            f.writelines(lines)

def readAllLines(path):
    with open(path, "r") as file:
        return file.readlines()

def loadData(path, seperator=" "):
    file = open(path, "r")
    data = file.readlines() 
    data = [line.replace("\n","") for line in data]
    data = [line.replace("__label__","") for line in data]
    return [line.split(seperator,1) for line in data] # todo: actually it should split on \t

def loadMillionPos(path):
    file = open(path, "r")
    data = file.readlines() 
    return [line.split("\t") for line in data]

def loadSb10k(path):
    file = open(path, "r")
    data = file.readlines() 
    data = [line.split("\t") for line in data] 
    data = [line for line in data if line[4] != 'Not Available\n'] # filter empty rows
    return [[line[1],line[4]] for line in data]

def loadGermeval2017(path):
    file = open(path, "r")
    data = file.readlines() 
    data = [line.split("\t") for line in data]     
    return [[line[3].strip(),line[1]] for line in data]

def loadSentimentLexicon(path):
    file = open(path, "r")
    data = file.readlines() 
    data = [line.split("\t") for line in data] 
    return [[line[3],line[0]] for line in data]


def loadImdb(path):
    file = open(path, "r")
    data = file.readlines() 
    data = [line.split("\",\"") for line in data] 
    for line in data:
        line[3] = "negative" if "NEG" in line[3] else "positive" 
    return [[line[3],line[2]] for line in data]    

def loadScareSet(directory):
    return [line for filePath in glob.glob(directory+"*.csv") for line in loadScare(filePath)]

def loadFilmstarts(path):
    file = open(path, "r")
    data = file.readlines() 
    data = [line.split("\t") for line in data] 

    # filter defect lines
    data = list(filter(lambda line:line[0].startswith("http://www.filmstarts.de") , data))

    for line in data:
        rateting = int(line[1])
        if rateting < 3:
            line[1] = "negative" 
        elif rateting > 3:
            line[1] = "positive"
        else:
            line[1] = "neutral"
            line[2] = ""

    return [[line[1],line[2]] for line in data]

def loadHolidaycheck(path):
    file = open(path, "r")
    data = file.readlines() 
    data = [line.split("\t") for line in data] 

    for line in data:
        rateting = int(line[0])
        if rateting < 3:
            line[0] = "negative" 
        elif rateting > 3:
            line[0] = "positive"
        else:
            line[0] = "neutral"
            line[1] = ""

    return [[line[0],line[1]] for line in data]

def loadScare(path):
    file = open(path, "r")
    data = file.readlines() 
    data = [line.split("\t") for line in data] 
    for line in data:
        rateting = int(line[1])
        if rateting < 3:
            line[1] = "negative" 
        elif rateting > 3:
            line[1] = "positive"
        else:
            line[1] = "neutral"
            line[2] = line[3] = ""

    return [[line[1],line[2] + " " + line[3]] for line in data] # include review text and headline in dataset
    #return [[line[1],line[2]] for line in data]
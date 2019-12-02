import re
from multiprocessing import Pool
from typing import List
import tools
from string import digits

from tqdm import tqdm


cleanChars = re.compile(r'[^A-Za-züöäÖÜÄß.!? ]', re.MULTILINE)
cleanHttpUrls = re.compile(r'https*\S+', re.MULTILINE)
cleanAtMentionsTwitter = re.compile(r'@\S+', re.MULTILINE)
        
config = tools.config()["preprocessing"]

def cleanData(data:List[str]) -> List[str]:
    with Pool(6) as p:        
        data = p.map(cleanRow, data)        

    #for row in tqdm(data):
    #    row[1] = cleanText(row[1])

    data = [row for row in data if row[1]] # filter out empty rows
    return data

def cleanRow(row):
    row[1] = cleanText(row[1])
    return row

def replaceNumbers(text: str) -> str:
        text = text.replace("0"," null")
        text = text.replace("1"," eins")
        text = text.replace("2"," zwei")
        text = text.replace("3"," drei")
        text = text.replace("4"," vier")
        text = text.replace("5"," fünf")
        text = text.replace("6"," sechs")
        text = text.replace("7"," sieben")
        text = text.replace("8"," acht")
        text = text.replace("9"," neun")
        return text

def loadSmileyData(path):
        file = open(path, "r")
        data = file.readlines()
        data = [line.replace("\n","") for line in data]
        return [line.split("\t") for line in data]        

simleys = loadSmileyData("../source-data/scare_v1.0.0_data/dictionaries/smiley.txt")
simleys.extend(loadSmileyData("../source-data/scare_v1.0.0_data/dictionaries/emoticons.txt"))

simleyChars =[smiley[0].strip() for smiley in simleys]

def containsSmiley(text: str) -> bool:
        for smiley in simleyChars:
                if smiley in text:
                        return True
        return False

def replaceSmiley(text):
        if containsSmiley(text):
                for simley in simleys:
                        text = text.replace(simley[0]," smiley" + simley[1] + " ")
        return text

def cleanText(text):              
        if(config["replace-smiley"] is True):
                text = replaceSmiley(text)
        
        text = text.replace("\n", " ")        
        text = cleanHttpUrls.sub('',text)
        text = cleanAtMentionsTwitter.sub('',text)

        if(config["replace-numbers-with-text"] is True):
                text = replaceNumbers(text)        
        
        text = cleanChars.sub('', text) ## use only text chars                          

        text = ' '.join(text.split()) # substitute multiple whitespace with single whitespace   
        text = text.strip()
        
        #text = lemmatize(text)
        #text = stripStopWords(text) # remove stopwords
        #text = stripCommonWords(text) # remove common
        
        # remove samples with a length over 1024
        # since they will be truncated by fasttext 
        # see MAX_LINE_SIZE in https://github.com/facebookresearch/fastText/blob/master/src/dictionary.h        
        return text[:config["max-line-length"]].lower()         

if __name__ == "__main__":    
#    sentence = stripStopWords("das ist ein toller satz")
#    print(sentence)

   # print(stripCommonWords("Dass du mir das schreibst muss wohl liebe sein"))


    #print(cleanText("@bunny du bist 1 nices gürl ▽ !!!1111einself https://nicees.gurl http://test@domain.org"))

    sentences = [
    ":-) test",
    "CANT WAIT for the new season of #TwinPeaks ＼(^o^)／!!! #davidlynch #tvseries :)))",
    "I saw the new #johndoe movie and it suuuuucks!!! WAISTED $10... #badmovies :/",
    "@SentimentSymp:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :-D http://sentimentsymposium.com/.",
    "@bunny du bist 1 nices gürl ▽ !!!1111einself https://nicees.gurl http://test@domain.org",
    "Dass du mir das schreibst muss wohl liebe sein",
    "“@sanitario_: Ich bin so Scheiße drauf, #ich #könnte #glatt #anfangen, #hashtags #zu #nutzen!” Dito!",
    "Das Essen war so gut.",
    "Der Fernseher ist so gut.",
    "der fernseher ist so gut."
    ]

    for s in sentences:
        print("{}\n----- ".format(cleanText(s)))
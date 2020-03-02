import fasttext
import tools
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
import printcm
from tabulate import tabulate
import glob
import os
from collections import defaultdict

class SystemUnderTest(object):
    """Base class for ml models for sequence classification data sets."""

    def predict(self, samples):
        """predicts the class for a list of known samples"""
        
        raise NotImplementedError()

class FastTextTest(SystemUnderTest):
    """FasText sequence classification tester"""
    def __init__(self,fastTextModel = None):
        if fastTextModel is None:
            fileType =  ".ftz" if tools.config()["model"]["quantize"] is True else ".bin"
            modelFile = tools.config()["model"]["model-path"] + fileType
            print("model under test:" + modelFile)
            self.model = fasttext.load_model(modelFile)
        else:
            self.model = fastTextModel        

    def predict(self, samples):
        """predicts the class for a list of known samples"""
        #print("testing set" + samples[0])
        data = samples[1]    
        truth = [row[0] for row in data]        
        texts = [row[1] for row in data]
        predictions = self.model.predict(texts)
        predictions = tools.flatmap(predictions[0])
        
        map_classes = lambda x:[item.replace("__label__","") for item in x]
        return map_classes(truth),map_classes(predictions)



def stat_fscore(truth, predicted):
    precisionMicro, recallMicro, fscoreMicro, _ = score(truth, predicted, average='micro')
    precisionMacro, recallMacro, fscoreMacro, _ = score(truth, predicted, average='macro')

    return [precisionMicro.item(), recallMicro.item(), fscoreMicro.item(), precisionMacro.item(), recallMacro.item(), fscoreMacro.item()]


def readTestData(path):
    data = [line.replace("\n","").split("\t") for line in open(path, "r").readlines()]
    #the data should look like this
    #data = [["set-a","postivive","text"],["set-b","postivive","text"],["set-a","postivive","text"],["set-c","postivive","text"]]

    # group by dataset
    res = defaultdict(list)
    for sample in data: 
        res[sample[0]].append(sample[1:])

    #reformat data
    result = [[k,v] for k,v in res.items()]
    
    #result.append(["all",tools.flatmap([v for _,v in res.items()])]) 
    
    return result

def run(fastTextModel, printErrors=False):        

    model = FastTextTest(fastTextModel)    
    
    data = readTestData("modeldata/model.test")
    
    table = []        

    all_truth = []
    all_prediction = []
    for row in data:
        truth, prediction = model.predict(row)
        result = stat_fscore(truth, prediction)        
        table.append([row[0]] + result)
        all_truth.extend(truth)
        all_prediction.extend(prediction)

    table.append(["all"] +stat_fscore(all_truth,all_prediction))
    
    sumRow = ["sum"]
    for col in range(1, len(table[0])):
        rowSum = sum(map(lambda x: x[col], table))
        sumRow.append(rowSum)
    table.append(sumRow)

    headers = ["file", "precisionMicro", "recallMicro",
               "fscoreMicro", "precisionMacro", "recallMacro", "fscoreMacro"]
    print(tabulate(table, headers, tablefmt="pipe", floatfmt=".4f"))

    title_description="FastText"

    plt = printcm.plot_confusion_matrix(all_truth, all_prediction, classes=[
                                      "negative", "neutral", "positive"], normalize=True, title=title_description)                                          
    plt.savefig("models/sentiment-cm.pdf")

    return table

if __name__ == "__main__":
    run(None, True)

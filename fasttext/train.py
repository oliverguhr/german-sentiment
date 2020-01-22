import os
from fasttext import train_supervised
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
import printcm
import yaml
import pprint
from tabulate import tabulate
import tools


def print_results(N, p, r):
    f1 = 2 *((p*r)/(p+r))
    print("N\t" + str(N))
    print("F1\t" + str(f1))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


def loadValidData(path):
    file = open(path, "r")
    data = file.readlines()
    data = [line.replace("\n", "") for line in data]
    return [line.split(" ", 1) for line in data]


def train(saveModel=True):
    train_data = tools.config()["model"]["train-file"]
    valid_data = tools.config()["model"]["valid-file"]
    quantizeModel=tools.config()["model"]["quantize"]    
    extendedValidation=tools.config()["model"]["print-confusion-matrix"]
    
    #quick train with hs
    #traningParameters = {'input': train_data, 'epoch': 10, 'lr': 0.25, 'wordNgrams': 3, 'verbose': 2, 'minCount':1, 'loss': "ns", "neg":5,
    #                    'lrUpdateRate': 100, 'thread': 8, 'ws': 5, 'dim': 100, 'pretrainedVectors': "model/sentiment.vector.d100.vec"}
    #traningParameters = {'input': train_data, 'epoch': 50, 'lr': 0.05, 'wordNgrams': 3, 'verbose': 2, 'minCount':1, 'loss': "ns",
    #                    'lrUpdateRate': 100, 'thread': 8, 'ws': 5, 'dim': 300, 'pretrainedVectors': "cc.de.300.vec"}                        
    traningParameters = tools.config()["model"]["fasttext"]
    traningParameters["input"] = train_data  

    pp = pprint.PrettyPrinter(depth=1)
    print("\n traing with following parameters ")
    pp.pprint(traningParameters)

    model = train_supervised(**traningParameters)

    if quantizeModel:
        print("quantize model")
        model.quantize(input=train_data,thread=16,qnorm=True,retrain=True,cutoff=400000) #

    if saveModel:
        path = tools.config()["model"]["model-path"]
        if quantizeModel:
            model.save_model(path+".ftz")
        else:
            model.save_model(path+".bin")        
        with open(path+".params", "w") as text_file:
            print(yaml.dump(tools.config()), file=text_file)  

    # validation
    if(extendedValidation is False):
        print_results(*model.test(valid_data))
    else:
        data = loadValidData(valid_data)

        truth = [row[0].replace("__label__", "") for row in data]        
        texts = [row[1] for row in data]

        predictions = model.predict(texts)
        predictions = tools.flatmap(predictions[0])
        predicted = [x.replace("__label__", "") for x in predictions]
 
        precision, recall, fscore, support = score(
            truth, predicted)  # , average='macro')


        headers = ["metric", "negative", "neutral","positive"] # todo: check if headers a right
        table = []
        table.append(['precision']+[x for x in precision])
        table.append(['recall']+[x for x in recall])
        table.append(['fscore']+[x for x in fscore])
        table.append(['sample count']+[x for x in support])
        print(tabulate(table, headers, tablefmt="pipe", floatfmt=".4f"))
  

        precision, recall, fscore, support = score(
            truth, predicted, average='macro')
        print('macro fscore: {}'.format(fscore))
        precision, recall, fscore, support = score(
            truth, predicted, average='micro')
        print('micro fscore: {}'.format(fscore))

        cm = confusion_matrix(truth, predicted, labels=[
                              "negative", "neutral", "positive"])
        printcm.plot_confusion_matrix(cm=cm, target_names=[
                                      "negative", "neutral", "positive"], normalize=True, title="sentiment classification")

    return model


if __name__ == "__main__":
    train(True)

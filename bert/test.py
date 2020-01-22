import os
import datetime
import logging

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from tqdm import tqdm
from tabulate import tabulate

import processors
import tools
import argparse
import printcm

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class Args(object):
    pass

class SystemUnderTest(object):
    """Base class for ml models for sequence classification data sets."""

    def predict(self, samples):
        """predicts the class for a list of known samples"""
        
        raise NotImplementedError()

class BertTest(SystemUnderTest):
    """FasText sequence classification tester"""
    def __init__(self,no_cuda,local_rank,task_name,output_dir,data_dir,bert_model,do_lower_case,max_seq_length,eval_batch_size,fp16):
        self.no_cuda = no_cuda
        self.local_rank = local_rank
        self.task_name = task_name
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.bert_model = bert_model
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.eval_batch_size = eval_batch_size
        self.fp16 = fp16

        self.processor = processors.processor_for_task(self.task_name)
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)

        model_file = os.path.join(args.output_dir, "pytorch_model.bin")        

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=self.do_lower_case)
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()

        model_state_dict = torch.load(model_file, map_location='cpu') if args.no_cuda else torch.load(model_file)
        self.model = BertForSequenceClassification.from_pretrained(self.bert_model, state_dict=model_state_dict, num_labels = self.num_labels)            
        self.model.to(self.device)
    
    @torch.no_grad()   
    def predict(self, eval_examples):
        """predicts the class for a list of known samples"""          
        eval_features = tools.convert_examples_to_features(eval_examples, self.label_list, self.max_seq_length, self.tokenizer)
        eval_examples_count = len(eval_examples)
        
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", eval_examples_count)
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        self.model.eval()

        if self.fp16:
            self.model.half()

        predictions = np.array([])
        truth = np.array([f.label_id for f in eval_features],dtype=int)

        for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="test batches"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            #label_ids = label_ids.to(device)
            
            #tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = self.model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            #label_ids = label_ids.numpy()        
            #eval_loss += tmp_eval_loss.mean().item()        
            predictions = np.append(predictions,np.argmax(logits, axis=1)) 

        # returning full text labels rather then numbers helps tracing errors
        lookup = self.processor.get_labels()
        map_classes = lambda x: [lookup[int(item)] for item in x]
        
        return map_classes(truth), map_classes(predictions)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def stat_fscore(truth, predicted):
    precisionMicro, recallMicro, fscoreMicro, _ = score(truth, predicted, average='micro')
    precisionMacro, recallMacro, fscoreMacro, _ = score(truth, predicted, average='macro')

    return [precisionMicro.item(), recallMicro.item(), fscoreMicro.item(), precisionMacro.item(), recallMacro.item(), fscoreMacro.item()]

def run(args):        

    model = BertTest(**vars(args))
    
    processor = processors.processor_for_task(args.task_name)
    data = processor.get_text_data_by_dataset(args.data_dir)
        
    table = []        

    all_truth = []
    all_prediction = []
    print("datasets")
    print([f" {row[0]} - {len(row[1])} " for row in data])
    
    for row in data:
        truth, prediction = model.predict(row[1])
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

    
    plt = printcm.plot_confusion_matrix(all_truth, all_prediction, classes=[
                                      "negative", "neutral", "positive"], normalize=True, title="Bert unbalanced")                                          
                                      
    plt.savefig(args.output_dir + "cm.pdf")

    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")    
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")                                                        
    args = parser.parse_args()
    
    run(args)
    


    #args = Args()
    #args.no_cuda = False
    #args.local_rank = -1
    #args.task_name = "sentiment"
    #args.output_dir ="tmp/sentiment-full-3-epoch-lm-20-uncased"
    #args.data_dir = "data/sentiment"
    #args.bert_model ="pretraining/finetuned_simple_lm_uncased_20"
    #args.do_lower_case = True
    #args.max_seq_length = 128
    #args.eval_batch_size = 100
    #    
    #test(args)


    #args = Args()
    #args.no_cuda = False
    #args.local_rank = -1
    #args.task_name = "offensive-language"
    #args.output_dir ="tmp/offensive-language-full-3-epoch-cased"
    #args.data_dir = "data/offensive-language"
    #args.bert_model ="bert-base-multilingual-cased"
    #args.do_lower_case = False
    #args.max_seq_length = 128
    #args.eval_batch_size = 128            

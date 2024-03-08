# This python file contains the code and functions required for running the basic and advanced models


# Importing utils and libraries
import time
import pandas as pd
import transformers
import numpy as np
from transformers import AutoTokenizer,AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_metric
from datasets import Dataset
from utils_final import read_data_as_sentence,map_labels_in_dataframe,tokenize_and_align_labels,get_label_mapping,get_labels_from_map,load_srl_model,load_dataset,compute_metrics,write_predictions_to_csv,compute_evaluation_metrics_from_csv
from sklearn.metrics import precision_recall_fscore_support

def define_args(mode='basic'):
    args=[]
    if mode == 'basic':
        arg1=['data/en_ewt-up-train.conllu', 'data/en_ewt-up-train.preprocessed_bas.csv']
        arg2=['data/en_ewt-up-dev.conllu', 'data/en_ewt-up-dev.preprocessed_bas.csv']
        arg3=['data/en_ewt-up-test.conllu', 'data/en_ewt-up-test.preprocessed_bas.csv']
        args.append(arg1)
        args.append(arg2)
        args.append(arg3)
    elif mode == 'advanced':
        arg1=['data/en_ewt-up-train.conllu', 'data/en_ewt-up-train.preprocessed_adv.csv']
        arg2=['data/en_ewt-up-dev.conllu', 'data/en_ewt-up-dev.preprocessed_adv.csv']
        arg3=['data/en_ewt-up-test.conllu', 'data/en_ewt-up-test.preprocessed_adv.csv']
    #     arg1=['/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/en_ewt-up-train.conllu', '/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/en_ewt-up-train.preprocessed_bas.csv']
    #     arg2=['/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/en_ewt-up-dev.conllu', '/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/en_ewt-up-dev.preprocessed_bas.csv']
    #     arg3=['/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/en_ewt-up-test.conllu', '/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/en_ewt-up-test.preprocessed_bas.csv']
    #     args.append(arg1)
    #     args.append(arg2)
    #     args.append(arg3)
    # elif mode == 'advanced':
    #     arg1=['/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/en_ewt-up-train.conllu', '/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/en_ewt-up-train.preprocessed_adv.csv']
    #     arg2=['/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/en_ewt-up-dev.conllu', '/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/en_ewt-up-dev.preprocessed_adv.csv']
    #     arg3=['/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/en_ewt-up-test.conllu', '/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/en_ewt-up-test.preprocessed_adv.csv']
        args.append(arg1)
        args.append(arg2)
        args.append(arg3)
    return args



def main(args, mode='basic', funct='train', results_path=None, data_range=None ):

    if funct =='train':

        if mode=='basic':
            train_data = read_data_as_sentence(args[0][0], args[0][1], mode='basic')
            dev_data = read_data_as_sentence(args[1][0], args[1][1],mode='basic')
            test_data = read_data_as_sentence(args[2][0], args[2][1],mode='basic')

        elif mode=='advanced':
            train_data = read_data_as_sentence(args[0][0], args[0][1], mode='advanced')
            dev_data = read_data_as_sentence(args[1][0], args[1][1],mode='advanced')
            test_data = read_data_as_sentence(args[2][0], args[2][1],mode='advanced')



        else:
            print('ERROR: Mode keyword parameter not found')

        model_checkpoint = "distilbert-base-uncased"


        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

        label_map = get_label_mapping(train_data, test_data, dev_data)


        train_data_mapped = map_labels_in_dataframe(train_data,label_map)
        dev_data_mapped = map_labels_in_dataframe(dev_data,label_map)
        test_data_mapped = map_labels_in_dataframe(test_data,label_map)

        tokenized_test = tokenize_and_align_labels(tokenizer, test_data_mapped, label_all_tokens=True)
        tokenized_train = tokenize_and_align_labels(tokenizer, train_data_mapped, label_all_tokens=True)
        tokenized_dev = tokenize_and_align_labels(tokenizer, dev_data_mapped, label_all_tokens=True)

        dataset_train = load_dataset(tokenized_train)
        dataset_dev = load_dataset(tokenized_dev)
        dataset_test = load_dataset(tokenized_test)

        if data_range != None:
            train_dataset = dataset_train.shuffle(seed=42).select(range(data_range))
            eval_dataset = dataset_dev.shuffle(seed=42).select(range(data_range))
            test_dataset = dataset_test.shuffle(seed=42).select(range(data_range))
        else:
            train_dataset = dataset_train.shuffle(seed=42)
            eval_dataset = dataset_dev.shuffle(seed=42)
            test_dataset = dataset_test.shuffle(seed=42)

        label_list = get_labels_from_map(label_map)


        model, model_name, args = load_srl_model(model_checkpoint, label_list)


        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda p: compute_metrics(*p, label_list))
        trainer.train()


        trainer.evaluate()


        predictions, labels, _ = trainer.predict(test_dataset)
        results = compute_metrics(predictions, labels, label_list)
        results

        # if mode=='basic':
        #     results_file = '/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/predictions_bas.csv'
        # elif mode == 'advanced':
        #     results_file = '/content/drive/My Drive/Colab Notebooks/Assignment3_nlp/data/predictions_adv.csv'
        if mode=='basic':
            results_file = 'data/predictions_bas.csv'
        elif mode == 'advanced':
            results_file = 'data/predictions_adv.csv'
        write_predictions_to_csv(predictions, labels, label_list, results_file, test_dataset['input_ids'][:data_range], tokenizer, test_data['input_form'][:data_range])
        classification_report = compute_evaluation_metrics_from_csv(results_file)
        print(classification_report)
    elif funct=='eval':
        results_file = results_path
        classification_report = compute_evaluation_metrics_from_csv(results_file)
        print(classification_report)

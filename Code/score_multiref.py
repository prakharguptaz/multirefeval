import os
import json
import copy
import csv
from random import randrange, shuffle
import glob
import argparse
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu
import nltk.translate.nist_score as nist_score
from sumeval.metrics.rouge import RougeCalculator
rouge = RougeCalculator(stopwords=True, lang="en")
import spacy
nlp = spacy.load('en')
from nlgeval import NLGEval
nlgeval = NLGEval(metrics_to_omit=['CIDEr']) 
cc = SmoothingFunction()

def clean_tokenize_sentence(data):
    data = data.lower().strip()#.replace(" \' ", "\'")
    spacy_token = nlp(data)
    
    if len(spacy_token)>0 and spacy_token[-1].text == 'eos':
        spacy_token = spacy_token[:-2]
    if len(spacy_token)>0 and spacy_token[0].text == '_':
        spacy_token = spacy_token[2:]

    if len(spacy_token)==0:
        return ['.']
    return [(token.text) for token in spacy_token]

def clean_split_sentence(data):
    data = data.lower().strip().replace('_go ', '').replace(' _eos', '')
    data_list = data.split()
    # return a . if no text left after cleanup
    if len(data_list)==0:
        return ['.']

    return data_list

def load_json(self, FILES_PATH, file = None):
    data = []
    with open(os.path.join(FILES_PATH, file)) as f:
        for line in f:
            data.append(json.loads(line))       
    #     print(data[1])

    return data

def read_duid_mapping_json(json_file):
    with open(json_file+'.json') as json_file:  
        data = json.load(json_file)
    
    return data

def read_predicted_data(pred_file):
    with open(pred_file) as fp:
        lines = fp.readlines()
    #shuffle(lines) # to test generic outputs
    lines = [clean_split_sentence(line) for line in lines]
    
    return lines

def read_predicted_data_asref(pred_file):
    with open(pred_file) as fp:
        lines = fp.readlines()
    #shuffle(lines) # to test generic outputs
    lines = [[clean_split_sentence(line)] for line in lines]
    
    return lines


def read_multiref_data(file_name):
    csv_data = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            csv_data.append(row)
            line_count += 1
        print(f'Processed {line_count} lines.')
        
    return csv_data

def read_multiref_premappeddata(file_name, num_response = -1):
    csv_data = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if num_response>0:
                row = row[:num_response]
            row_tokenized = [item.split() for item in row]
            csv_data.append(row_tokenized)
            line_count += 1
        print(f'Processed {line_count} lines, premapped file.')
        
    return csv_data

def read_multiref_data_hyp(file_name):
    csv_data = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            row = [row.split('||||')[0]]
            csv_data.append(row)
            line_count += 1
        print(f'Processed {line_count} lines.')
        
    return csv_data

def load_json_file(file = None):
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))       

    return data

def get_ref_hyp_pairs_json(multiref_data, predictions, prevgt_ref, mapping_json, num_response = -1):

    list_ref, list_hypothesis, list_prevgt_ref = [], [], []
    count_processed = 0
    break_done = False
    for did, _ in enumerate(multiref_data):
        #did and dialogue_id shoud be same
        dialogue_id = multiref_data[did]['index']
        if break_done:
            break
        for u, utterance_data in enumerate(multiref_data[dialogue_id]['dialogue']):
            #skip the last utterance as it does not have a response
            if u == len(multiref_data[dialogue_id]['dialogue'])-1:
                break
            dialogue_utterance_id = str(dialogue_id) + '_' + str(u)
            references = utterance_data['responses']
            references = [clean_split_sentence(line) for line in references]
            if num_response>0:
                references = references[:num_response]
            line_number = mapping_json[dialogue_utterance_id]
            ## prediction file already cleaned and tokenised
            prediction = predictions[line_number-1]
            prev_gt = prevgt_ref[line_number-1]
            ## remove _eos
            # if prediction[-1] == 'eos':
            #     prediction = prediction[:-2]
            
            list_ref.append(references)
            list_hypothesis.append(prediction)
            list_prevgt_ref.append(prev_gt)
            count_processed += 1
        
    return list_ref, list_hypothesis, list_prevgt_ref

def get_ref_hyp_pairs(multiref_data, predictions, prevgt_ref, mapping_json, num_response = -1):
    list_ref, list_hypothesis, list_prevgt_ref = [], [], []
    
    for i, data in enumerate(multiref_data):
        dialogue_utterance_id = data[0]
        references = data[3:num_response+3]
        if num_response>0:
            references = references[:num_response]
        references = [clean_split_sentence(line) for line in references]

        line_number = mapping_json[dialogue_utterance_id]
        ## prediction file already cleaned and tokenised
        prediction = predictions[line_number-1]
        prev_gt = prevgt_ref[line_number-1]
        ## remove _eos
        # if prediction[-1] == 'eos':
        #     prediction = prediction[:-2]
        
        list_ref.append(references)
        list_hypothesis.append(prediction)
        list_prevgt_ref.append(prev_gt)
        if i == 100:
            break

    return list_ref, list_hypothesis, list_prevgt_ref


def get_ref_hyp_pairs_prevgtref(multiref_data, prevgt_ref, mapping_json):
    '''
    for parsing muktiref data saved in csv format
    '''
    list_ref, list_hypothesis = [], []
    
    for i, data in enumerate(multiref_data):
        dialogue_utterance_id = data[0]
        hypothesis = clean_split_sentence(data[1].split('||||')[0])
        # references = [clean_tokenize_sentence(line) for line in references]
        line_number = mapping_json[dialogue_utterance_id]
        ## prediction file already cleaned and tokenised
        prevgt_ref_list = prevgt_ref[line_number-1]
        ## remove _eos
        if hypothesis[-1] == 'eos':
            hypothesis = hypothesis[:-2]
        
        list_ref.append(prevgt_ref_list)
        list_hypothesis.append(hypothesis)

    return list_ref, list_hypothesis


def get_rouge(hypothesis_tokens, references_tokens):
    hypothesis = (' ').join(hypothesis_tokens)
    references = [(' ').join(references_t) for references_t in references_tokens]
    rouge_1 = rouge.rouge_n(
            summary=hypothesis,
            references=references,
            n=1)

    rouge_2 = rouge.rouge_n(
            summary=hypothesis,
            references=references,
                n=2)

    rouge_l = rouge.rouge_l(
            summary=hypothesis,
            references=references)
    return rouge_1, rouge_2, rouge_l

def calculate_rouge_data(list_hypothesis, list_references):

    rouge_1_list, rouge_2_list, rouge_l_list = [], [], []  
    for i,_ in enumerate(list_references):
        rouge_1, rouge_2, rouge_l = get_rouge(list_hypothesis[i], list_references[i])
        rouge_1_list.append(rouge_1)
        rouge_2_list.append(rouge_2)
        rouge_l_list.append(rouge_l)

    avg_rouge_1 = sum(rouge_1_list)/len(rouge_1_list)
    avg_rouge_2 = sum(rouge_2_list)/len(rouge_2_list)
    avg_rouge_l = sum(rouge_l_list)/len(rouge_l_list)

    return avg_rouge_1, avg_rouge_2, avg_rouge_l

def calculate_maxrouge_data(list_hypothesis, list_references):

    rouge_1_list, rouge_2_list, rouge_l_list = [], [], []  
    for i,_ in enumerate(list_references):
        rouge1_max_list, rouge2_max_list, rougel_max_list = [], [], []
        for item in list_references[i]:
            rouge_1_item, rouge_2_item, rouge_l_item = get_rouge(list_hypothesis[i], [item])
            rouge1_max_list.append(rouge_1_item)
            rouge2_max_list.append(rouge_2_item)
            rougel_max_list.append(rouge_l_item)
        rouge_1 = max(rouge1_max_list)
        rouge_2 = max(rouge2_max_list)
        rouge_l = max(rougel_max_list)

        rouge_1_list.append(rouge_1)
        rouge_2_list.append(rouge_2)
        rouge_l_list.append(rouge_l)

    avg_rouge_1 = sum(rouge_1_list)/len(rouge_1_list)
    avg_rouge_2 = sum(rouge_2_list)/len(rouge_2_list)
    avg_rouge_l = sum(rouge_l_list)/len(rouge_l_list)

    return avg_rouge_1, avg_rouge_2, avg_rouge_l

def calculate_max_bleu(list_references, list_hypothesis, weights):
    sum_bleu = 0.0
    for i,d in enumerate(list_references):
        references_items = list_references[i]
        hypothesis = list_hypothesis[i]
        bleu_score_sentence = []
        for reference in references_items:
            bleu_score_sentence.append(sentence_bleu([reference], hypothesis, weights, smoothing_function=cc.method1))
        sum_bleu += max(bleu_score_sentence)
    mean_bleu = sum_bleu / len(list_hypothesis)   

    return mean_bleu

def calculate_max_nist(list_references, list_hypothesis):
    sum_bleu = 0.0
    for i,d in enumerate(list_references):
        references_items = list_references[i]
        hypothesis = list_hypothesis[i]
        bleu_score_sentence = []
        for reference in references_items:
            try:
                bleu_score_sentence.append(nist_score.sentence_nist([reference], hypothesis))
                sum_bleu += max(bleu_score_sentence)

            except:
                sum_bleu += 0
    mean_bleu = sum_bleu / len(list_hypothesis)   

    return mean_bleu



def calculate_sentence_bleu(list_references, list_hypothesis, weights):
    sum_bleu = 0.0
    for i,d in enumerate(list_references):
        references_items = list_references[i]
        hypothesis = list_hypothesis[i]
        bleu_score_sentence = []
        sum_bleu+=sentence_bleu(references_items, hypothesis, weights)
        # for reference in references_items:
        #     bleu_score_sentence.append(sentence_bleu([reference], hypothesis, weights))
        # sum_bleu += sum(bleu_score_sentence)/len(bleu_score_sentence)
    mean_bleu = sum_bleu / len(list_hypothesis)   

    return mean_bleu

def get_sentence_bleu(list_references, list_hypothesis):
    score_corpusBLEU1 = calculate_sentence_bleu(list_references, list_hypothesis,  weights =(1.0, 0.0, 0.0, 0.0))    
    score_corpusBLEU2 = calculate_sentence_bleu(list_references, list_hypothesis,  weights =(0.5, 0.5, 0.0, 0.0))    
    score_corpusBLEU4 = calculate_sentence_bleu(list_references, list_hypothesis,  weights =(0.25, 0.25, 0.25, 0.25))   
    
    print('\nAverage sentence bleu-1;'+'\t', score_corpusBLEU1)
    print('Average sentence bleu-2;'+'\t', score_corpusBLEU2)
    print('Average sentence bleu-4;'+'\t', score_corpusBLEU4)

def get_max_nist(list_references, list_hypothesis):
    score_nist = calculate_max_nist(list_references, list_hypothesis)   
    
    print('\nAverage max sentence nist;'+'\t', score_nist)

def get_max_bleu(list_references, list_hypothesis):
    score_corpusBLEU1 = calculate_max_bleu(list_references, list_hypothesis,  weights =(1.0, 0.0, 0.0, 0.0))    
    score_corpusBLEU2 = calculate_max_bleu(list_references, list_hypothesis,  weights =(0.5, 0.5, 0.0, 0.0))
    score_corpusBLEU3 = calculate_max_bleu(list_references, list_hypothesis,  weights =(0.33, 0.33, 0.33, 0.0))        
    score_corpusBLEU4 = calculate_max_bleu(list_references, list_hypothesis,  weights =(0.25, 0.25, 0.25, 0.25))   
    
    print('\nAverage max sentence bleu-1;'+'\t', score_corpusBLEU1)
    print('Average max sentence bleu-2;'+'\t', score_corpusBLEU2)
    print('Average max sentence bleu-3;'+'\t', score_corpusBLEU3)
    print('Average max sentence bleu-4;'+'\t', score_corpusBLEU4)

def get_rouge_scores(list_references, list_hypothesis):
    avg_rouge_1, avg_rouge_2, avg_rouge_l = calculate_rouge_data(list_hypothesis, list_references)
    print('Average avg_rouge_1;'+'\t', avg_rouge_1)
    print('Average avg_rouge_2;'+'\t', avg_rouge_2)
    print('Average avg_rouge_l;'+'\t', avg_rouge_l)

def get_maxrouge_scores(list_references, list_hypothesis):
    avg_rouge_1, avg_rouge_2, avg_rouge_l = calculate_maxrouge_data(list_hypothesis, list_references)
    print('Average max avg_rouge_1;'+'\t', avg_rouge_1)
    print('Average max avg_rouge_2;'+'\t', avg_rouge_2)
    print('Average max avg_rouge_l;'+'\t', avg_rouge_l)


def get_corpus_bleu(list_references, list_hypothesis):
    score_corpusBLEU1 = corpus_bleu(list_references, list_hypothesis,  weights =(1.0, 0.0, 0.0, 0.0))    
    score_corpusBLEU2 = corpus_bleu(list_references, list_hypothesis,  weights =(0.5, 0.5, 0.0, 0.0))    
    score_corpusBLEU4 = corpus_bleu(list_references, list_hypothesis,  weights =(0.25, 0.25, 0.25, 0.25))   
    
    print('\nAverage corpus bleu-1;'+'\t', score_corpusBLEU1)
    print('Average corpus bleu-2;'+'\t', score_corpusBLEU2)
    print('Average corpus bleu-4;'+'\t', score_corpusBLEU4)



def get_metrics_multiref_frompremapped(args):
    list_references = read_multiref_premappeddata(args.premappedmulti_csv_file, num_response = args.num_multi_response)
    list_hypothesis = read_predicted_data(args.pred_file)
    
    # pdb.set_trace()
    get_all_metrics(list_references, list_hypothesis)

    return list_references, list_hypothesis

def get_metrics_frompremapped_prevgt(args):
    print("\n-Metrics with previous ground truth-")
    list_references = read_predicted_data(args.singleref_file)
    list_hypothesis = read_predicted_data(args.pred_file)
    
    list_references_listfied = [[ref] for ref in list_references]

    # pdb.set_trace()
    get_all_metrics(list_references_listfied, list_hypothesis)

    return list_references, list_hypothesis


def get_metrics_from_multirefasmodel_prevgt(args):
    print("Calculating collected data's 1st response score with prev ground truth")
    multiref_data = read_multiref_data(args.csv_file)
    prevgt_ref = read_predicted_data_asref(args.singleref_file)
    mapping_json = read_duid_mapping_json(args.fold+ '_duid_mapping')
    
    list_references, list_hypothesis = get_ref_hyp_pairs_prevgtref(multiref_data, prevgt_ref, mapping_json)

    get_all_metrics(list_references, list_hypothesis)

    return list_references, list_hypothesis


def get_metrics_multiref_frommapping(args):
    print("Metrics with Multi-ref ground truth")
    multiref_data = load_json_file(args.multiref_file)
    predictions = read_predicted_data(args.pred_file)
    prevgt_ref = read_predicted_data_asref(args.singleref_file)

    # mapping each line in test file to correct contextid
    mapping_json = read_duid_mapping_json(args.fold+ '_duid_mapping')
    print('reading files complete')
    list_references, list_hypothesis, list_prev_gt = get_ref_hyp_pairs_json(multiref_data, predictions, prevgt_ref, mapping_json, num_response = args.num_multi_response)
    get_all_metrics(list_references, list_hypothesis)


    return list_references, list_hypothesis

def add_prevgt_to_multiref(list_references, list_hypothesis, index_to_replace = 0):
    print('\n\t\t\tAdding gt to multiref')
    ref = [s + m for s,m in zip(list_hypothesis, list_references)]

    return ref, list_references

def convert_tostring_lists(list_references, list_hypothesis):
    list_string_references, list_string_hypothesis = [], []
    for hypothesis_itemlist in list_hypothesis:
        list_string_hypothesis.append(' '.join(hypothesis_itemlist))

    for references_list in list_references:
        list_string_references.append([' '.join(individual_ref) for individual_ref in references_list])

    ##convert references to n separate lists of referenes
    num_responses = len(list_string_references[0])
    mod_reference_list = [[] for i in range(num_responses)]
    for i, item in enumerate(list_string_references):
        for j, ref in enumerate(item):
            mod_reference_list[j].append(ref)

    return mod_reference_list, list_string_hypothesis

def get_max_avg_metrics(list_references, list_hypothesis):
    sum_bleu, sum_rouge, sum_skip, sum_ae, sum_ve, sum_ge = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i,d in enumerate(list_references):
        references_items = list_references[i]
        hypothesis = list_hypothesis[i]
        hypothesis = ' '.join(hypothesis)
        bleu_score_sentence, rougel_score_sentence, SkipThoughtCS_score_sentence, AE_score_sentence, VE_score_sentence, GE_score_sentence  = [], [], [], [], [], []
        if i%1000 ==0:
            print(i)
        for reference in references_items:
            reference = ' '.join(reference)
            scores = nlgeval.compute_individual_metrics(ref=[reference], hyp=hypothesis)
            bleu_score_sentence.append(scores['METEOR'])
            rougel_score_sentence.append(scores['ROUGE_L'])
            SkipThoughtCS_score_sentence.append(scores['SkipThoughtCS'])
            AE_score_sentence.append(scores['EmbeddingAverageCosineSimilairty'])
            VE_score_sentence.append(scores['VectorExtremaCosineSimilarity'])
            GE_score_sentence.append(scores['GreedyMatchingScore'])
            # print(hypothesis, reference, scores)
        sum_bleu += max(bleu_score_sentence)
        sum_rouge += max(rougel_score_sentence)
        sum_skip += max(SkipThoughtCS_score_sentence)
        sum_ae += max(AE_score_sentence)
        sum_ve += max(VE_score_sentence)
        sum_ge += max(GE_score_sentence)

    mean_bleu = sum_bleu / len(list_hypothesis)
    mean_rouge = sum_rouge / len(list_hypothesis)  
    mean_skipthoughtcs = sum_skip / len(list_hypothesis)  
    mean_ae = sum_ae / len(list_hypothesis)  
    mean_ve = sum_ve / len(list_hypothesis)  
    mean_ge = sum_ge / len(list_hypothesis)   

    return mean_bleu, mean_rouge, mean_skipthoughtcs, mean_ae, mean_ve, mean_ge


def print_metrics_dict(metrics_dict):
    for metric in metrics_dict.keys():
        print(metric + ';\t' + str(metrics_dict[metric]))

def get_all_metrics(list_references, list_hypothesis):

    # get_corpus_bleu(list_references, list_hypothesis)
    # get_sentence_bleu(list_references, list_hypothesis)
    get_max_bleu(list_references, list_hypothesis)

    list_string_references, list_string_hypothesis = convert_tostring_lists(list_references, list_hypothesis)
    metrics_dict = nlgeval.compute_metrics(list_string_references, list_string_hypothesis)
    print_metrics_dict(metrics_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiref_file', default='../multiref-dataset/multireftest.json')
    parser.add_argument('--singleref_file', default='jsons/test.tgt')
    parser.add_argument('--pred_file', default='')
    parser.add_argument('--fold', default='jsons/test')
    parser.add_argument('--num_multi_response', default=5)
    args = parser.parse_args()
    
    ''''''
    # get_metrics_multiref_frompremapped(args)
    print('******Testing model*******')
    # ## test using mutli ref
    get_metrics_multiref_frommapping(args)

    ## test using single ref
    get_metrics_frompremapped_prevgt(args)

if __name__ == "__main__":
    main()                

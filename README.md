# multirefeval
Data and evaluation scipt for the paper "Investigating Evaluation of Open-Domain Dialogue Systems With Human Generated Multiple References" in SIGdial 2019

## Data
- Data is present in the folder `multiref-dataset`
- It is formatted in a json format. The file contains 1000 jsons, 1 on each line, corresponding to a test dialog form the dailydialog dataset.
- Every dialog json contains the dialog text in the 'dialogue' key field. The value for 'dialogue' key field is a list of utterances jsons.
- In each utterance json, the multiple references are present in the responses field. The first response corresponds to the original reference from the dataset and is the same as the next utterance text in the dialog.
 -Since the last utterance in a dialog doesn't have a reply, it does not have a responses field.

## Code 
We have provided evaluation script to use for multi-reference and single reference evaluation using the data above. The code is present in the `Code` folder.

### Dependencies-
The code has following package dependencies -\
sumeval - pip install sumeval\
Maluuba's nlgeval - [link](https://github.com/Maluuba/nlg-eval)

### Files
score_multiref.py - Code to run multi-reference evaluation\
hredf.txt - Sample model output file\
jsons/test.tgt - Single reference file\
jsons/test_duid_mapping.json - File contains mapping from context id to line number (more details below)

### Line number - context id mapping
The test dataset consists of 1000 dialogues, which leads to 6740 context-reply pairs. For e.g., if a dialog has 10 utterances, it corresponds to 9 context-reply pairs (the last utterance does not lead to a reply).\
Any model generated outputs will have 6740 lines corresponding to the contexts, as is the case of the sample hredf.txt file. 
We map the context id which is the dialog id concatenated with an utterance id to the lines in test output file using a python dictionary present in test_duid_mapping.json file

### Running evaluation script
For evaluation, you can use the command 
`python score_multiref.py  --pred_file hredf.txt`
The arguments for this file are following - 
 - --multiref_file - The multiref dataset file. Default is ../multiref-dataset/multireftest.json
 - --singleref_file - The single reference test file. Default is jsons/test.tgt
 - --pred_file - The file you want to run evaluation on
 - --num_multi_response - The number of references you want to use, Default=5

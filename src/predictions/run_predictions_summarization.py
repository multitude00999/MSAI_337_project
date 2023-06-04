from datasets import load_dataset,get_dataset_split_names,load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import pipeline
from metrics import calcSBERTScore,loadModel
import os
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import pickle
from typing import List
import yaml
import time

config_file = os.environ.get('CONFIG_FILE')
config_path = "../config/{}.yaml".format(config_file)
with open('/Users/igautam/Documents/GitHub/MSAI_337_project/src/config/exp_summ_CNN.yaml', "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

dataset = 'summeval'
# Set the path to save the dataset
dataset_path = config['text_summarization'][dataset]['dataset_path']
dataset_split = config['text_summarization'][dataset]['dataset_split']
dataset_config = config['text_summarization'][dataset]['dataset_config']
dataset_name = config['text_summarization'][dataset]['dataset_name']
dataset_input = config['text_summarization'][dataset]['dataset_input']
dataset_ground_truth = config['text_summarization'][dataset]['dataset_ground_truth']
dataset_praportion = config['text_summarization'][dataset]['dataset_praportion']

results_path = config['text_summarization'][dataset]['results_path']
results_filename = config['text_summarization'][dataset]['results_filename']


if not os.path.isdir(results_path):
  os.mkdir(results_path)

rouge = evaluate.load('rouge')
bertscore = evaluate.load("bertscore")
modelsbert_path = 'paraphrase-MiniLM-L6-v2'


# Check if the dataset exists in the specified path
if not os.path.exists(dataset_path):
    # Download the dataset if it doesn't exist
    dataset = load_dataset(dataset_name,dataset_config,split=dataset_split)
    dataset.save_to_disk(dataset_path)
else:
    # Load the dataset from the existing path
    print(dataset_path)
    dataset = load_from_disk(dataset_path)
    #dataset = load_dataset("path", data_dir=dataset_path)

print(int(len(dataset)*dataset_praportion))
mname = "google/pegasus-xsum"
model = PegasusForConditionalGeneration.from_pretrained(mname)
tok = PegasusTokenizer.from_pretrained(mname)

if(dataset == 'CNN'):
    inputs = tok(dataset[dataset_split][dataset_input][:int(len(dataset)*dataset_praportion)], truncation=True, padding="longest", return_tensors="pt")
    ground_truths = dataset[dataset_split][dataset_ground_truth][:int(len(dataset)*dataset_praportion)]
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)
    predicted_summary = tok.batch_decode(outputs, skip_special_tokens=True)
else:
    inputs = []
    inputs_seq = dataset[dataset_input][:int(len(dataset)*dataset_praportion)]
    for seq in inputs_seq:
        inputs.append(inputs_seq[0])

    inputs = tok(inputs, truncation=True, padding="longest", return_tensors="pt")
    ground_truths = dataset[dataset_ground_truth][:int(len(dataset)*dataset_praportion)]
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)
    predicted_summary = tok.batch_decode(outputs, skip_special_tokens=True)

# Measure time for results_rouge
start_time = time.time()
results_rouge = rouge.compute(predictions=predicted_summary, references=ground_truths)
end_time = time.time()
results_rouge_time = end_time - start_time

# Measure time for results_bertscore
start_time = time.time()
results_bertscore = bertscore.compute(predictions=predicted_summary, references=ground_truths, lang="en")
end_time = time.time()
results_bertscore_time = end_time - start_time

# Measure time for results_sBERT
modelsbert = loadModel(modelsbert_path)
start_time = time.time()
results_sBERT = calcSBERTScore(modelsbert, predicted_summary, ground_truths)
end_time = time.time()
results_sBERT_time = end_time - start_time

# Print the time taken for each prediction
print("Time taken for results_rouge: ", results_rouge_time, " seconds")
print("Time taken for results_bertscore: ", results_bertscore_time, " seconds")
print("Time taken for results_sBERT: ", results_sBERT_time, " seconds")

results = {
    'results_rouge':[results_rouge,results_rouge_time],
    'results_bertscore':[results_bertscore,results_bertscore_time],
    'results_sBERT':[results_sBERT,results_sBERT_time]
}


with open(results_path+results_filename, "wb") as file:
    pickle.dump(results,file)




from datasets import load_dataset,get_dataset_split_names,load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from metrics import calcSBERTScore,loadModel
from transformers import pipeline
from transformers import AutoModelWithLMHead, AutoTokenizer
#from metrics import calcSBERTScore,loadModel
import os
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import pickle
from typing import List
import yaml
import time


rouge = evaluate.load('rouge')
bertscore = evaluate.load("bertscore")
modelsbert_path = 'paraphrase-MiniLM-L6-v2'

dataset_path = '../../raw/wmt18'

# Check if the dataset exists in the specified path
if not os.path.exists(dataset_path):
    # Download the dataset if it doesn't exist
    dataset = load_dataset("wmt18",'cs-en','validation')
else:
    dataset = load_from_disk(dataset_path)


inputs = list(dataset['validation']['translation'][0:10])
inputs_cs = []
ground_truth_en = []
for inp in inputs:
  inputs_cs.append(str(inp['cs']))
  ground_truth_en.append(str(inp['en']))

outputs_en = []

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelWithLMHead.from_pretrained("t5-base")


for input_text in inputs_cs:
    inputs = tokenizer.encode_plus(input_text, return_tensors="pt", padding=True, truncation=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=40, num_beams=4, early_stopping=True)
    outputs_en.append(output)


# Measure time for results_rouge
start_time = time.time()
rouge = evaluate.load('rouge')
results_rouge = rouge.compute(predictions=outputs_en, references=ground_truth_en)
end_time = time.time()
results_rouge_time = end_time - start_time

# Measure time for results_bertscore
start_time = time.time()
bertscore = evaluate.load("bertscore")
results_bertscore = bertscore.compute(predictions=outputs_en, references=ground_truth_en, lang="en")
end_time = time.time()
results_bertscore_time = end_time - start_time

# Measure time for results_sBERT
modelsbert = loadModel(modelsbert_path)
start_time = time.time()
results_sBERT = calcSBERTScore(modelsbert, outputs_en, ground_truth_en)
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

with open('../../results/WMT18/results_wmt18', "wb") as file:
    pickle.dump(results,file)




from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
from datasets import load_dataset,get_dataset_split_names,load_from_disk
import yaml
from metrics import calcSBERTScore,bert_score,rougue,loadModel
import os
from transformers import T5Tokenizer, T5Model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
nltk.download('punkt')
     
'''
config_file = os.environ.get('CONFIG_FILE')
config_path = "../config/{}.yaml".format(config_file)
with open(config_path, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

if not os.path.isdir('../../data'):
  os.mkdir('../../data')

if not os.path.isdir('../../data/raw'):
  os.mkdir('../../data/raw')

if not os.path.isdir('../../data/processed'):
  os.mkdir('../../data/processed')
'''

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    model_path = 'paraphrase-MiniLM-L6-v2'
    model = loadModel(model_path)

    result_sBERT_score = calcSBERTScore(model,predictions=decoded_preds, references=decoded_labels,debug=False)
    result_BERT_score = bert_score(decoded_preds,decoded_labels)
    result_ROGUE = rougue(decoded_preds,decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# Set the path to save the dataset
dataset_path = "/Users/igautam/Documents/GitHub/MSAI_337_project/data/wmt18/raw"

# Check if the dataset exists in the specified path
if not os.path.exists(dataset_path):
    # Download the dataset if it doesn't exist
    dataset = load_dataset("wmt18","cs-en")
    dataset.save_to_disk(dataset_path)
else:
    # Load the dataset from the existing path
    print(dataset_path)
    dataset = load_from_disk(dataset_path)
    #dataset = load_dataset("path", data_dir=dataset_path)

# Access the dataset
train_dataset = dataset["train"]
test_dataset = dataset["test"]

print(len(train_dataset),len(test_dataset))

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

tokenized_books = train_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)









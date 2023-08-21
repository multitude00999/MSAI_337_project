# sBERT score: Evaluating text generation using Sentence BERT

This repository consists the code of final project of course [MSAI 337](https://www.mccormick.northwestern.edu/artificial-intelligence/curriculum/descriptions/msai-337.html) at Northwestern Univeristy in Spring 2023 taught by Dr. David Demeter. 

## Abstract
Text generation is a challenging task that requires evaluating the quality of generated text. Existing automatic metrics, such as ROUGE and BLEU scores, lack semantic understanding and struggle with paraphrased text. To address these limitations, semantic evaluation metrics like BERTScore have been proposed, but their computational cost remains a challenge. In this paper, we introduce sBERTScore, a novel evaluation metric inspired by BERTScore, aimed at reducing computational time while maintaining performance. We evaluate sBERTScore on machine translation and summarization tasks using standard datasets, comparing it with exact string matching metrics like ROGUE Score and semantic metrics like BERTScore. Our results demonstrate the effectiveness of sBERTScore in capturing semantic similarities between generated and reference text, making it a promising evaluation metric for text generation.

Project report available [here](sBERT_score.pdf)
## Results

|Model | relevance|  coherence |  consistency | fluency|
|:---: | :---: |  :---: | :---: | :---:|
|BERT-prec | 0.3407 |  0.3206 | 0.1729 | 0.2256|
|BERT-recall |0.3408 |0.2755 |0.1548 |0.2384|
|BERT-F1 |0.3764 |0.3302 |0.1804 |0.2563|
|ROUGE1| 0.2816| 0.2253| 0.1677| 0.1342|
|ROUGE2| 0.2146| 0.1809| 0.0502| 0.1576|
|ROUGEL| 0.2893| 0.2972| 0.1467| 0.0901|
|sBERTScore Prec| 0.3008| 0.3462| 0.1912| 0.2110|
|sBERTScore Recall| 0.2785| 0.2380| 0.1759| 0.1256|
|sBERTScore F1 |0.3087| 0.3161| 0.1903| 0.1796|
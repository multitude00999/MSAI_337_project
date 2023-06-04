# imports
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np

nltk.download('punkt')

def loadModel(model_path):
  return SentenceTransformer(model_path)

def getSentences(text):
  return sent_tokenize(text)

def getEmbeddings(model, sentences):
  embeddings = model.encode(sentences)
  return embeddings

def getSimilarityMatrix(generated_embs, reference_embs):
  return cosine_similarity(generated_embs, reference_embs)

def getScore(sim_mat):
  
  # get maximum along reference text axis
  scores = np.max(sim_mat, axis = 0)

  # calculate mean
  final_score = np.mean(scores)

  return final_score

def calcSBERTScore(model, generated_texts, reference_texts, debug = False):

  score_final = 0

  # get sentences
  for i in range(len(generated_texts)):

    generated_text = generated_texts[i]
    reference_text = reference_texts[i]

    generated_sents = getSentences(generated_text)
    reference_sents = getSentences(reference_text)

    # get embeddings
    generated_embs = getEmbeddings(model, generated_sents)
    reference_embs = getEmbeddings(model, reference_sents)

    # calculate pairwise cosine similarity
    sim_mat = cosine_similarity(generated_embs, reference_embs)

    if debug:
      print("sim mat shape", sim_mat.shape)
      print(sim_mat)
    # get score
    score = getScore(sim_mat)
    score_final = score_final + score
  
  return score_final/len(generated_texts)


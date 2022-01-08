# Similarity score calculation of keywords with explanations

This repository contains the codebase for getting similarity scores between a set of keywords with a paragraph explanation. SBERT.net sentence transformer model has been used to get the vector embedding for each keyword using the paragraph explanations. Then the calculated vector embeddings have been used to get the cosine similarities to calculate a similarity score between the embeddings.

For generating a dense embedding of each paragraph, two models were compared considering the speed and quality of the resultant embedding.

Out of all the general purpose models where trained on all available training data (more than 1 billion training pairs); for faster operation with good quality, all-MiniLM-L6-v2  model has been selected. For getting the best quality results, all-mpnet-base-v2 model can be used with 5 times slower speed compared to the previous model.

## Background of the all-MiniLM-L6-v2 model

all-MiniLM-L6-v2 sentence-transformer model maps sentences & paragraphs to a 384 dimensional dense vector space. This was pre trained using microsoft/MiniLM-L12-H384-uncased model and fine-tuned in on a 1B sentence pairs dataset. By default, input text longer than 256 word pieces is truncated.
Fine tuning of this model has been done by computing cosine similarity of each possible sentence pair from the batch and appling cross entropy loss compared with true pairs.

## Usage (Sentence-Transformers)
```
#Using this model becomes easy when you have sentence-transformers installed:
pip install -U sentence-transformers

#Then you can use the model like this:
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
embeddings = model.encode(sentences)
print(embeddings)
```


## Features
 - Description:	All-round model tuned for many use-cases. Trained on a large and diverse dataset of over 1 billion training pairs.
 - Base Model:	microsoft/MiniLM-L12-H384-uncased
 - Max Sequence Length:	256
 - Dimensions:	384
 - Normalized Embeddings:	true
 - Suitable Score Functions:	dot-product (util.dot_score), cosine-similarity (util.cos_sim), euclidean distance
 - Size:	118 MB
 - Pooling:	Mean Pooling
 - Training Data:	1B+ training pairs. For details, see model card.
 - Model Card:	https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2




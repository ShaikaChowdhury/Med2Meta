# Med2Meta

This repository contains the code for the paper,

Shaika Chowdhury, Chenwei Zhang, Philip S. Yu and Yuan Luo. Med2Meta: Learning Representations of Medical Concepts with Meta-Embeddings. BIOSTEC HEALTHINF 2020. 


## Usage

<dem_emb_file>: the file containing the graph embeddings wrt dem view

<lab_emb_file>: the file containing the graph embeddings wrt lab view

<notes_emb_file>: the file containing the graph embeddings wrt notes view

<_dims>: the embedding dimension
  
<_bs>: the batch size

run the following command:

python run.py -m M2M -i <dem_emb_file> <lab_emb_file> <notes_emb_file> -d <_dims> <_dims> <_dims> -o res.txt -b <_bs>

reference: https://github.com/LivNLP/AEME

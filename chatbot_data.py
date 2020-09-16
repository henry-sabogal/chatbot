#%%
import numpy as np
import pandas as pd
import json
import gensim
import gensim.parsing.preprocessing as preprocessing
import os
import collections
import random

from gensim.parsing.preprocessing import remove_stopwords, strip_tags, strip_punctuation, strip_numeric, stem_text, strip_short
from gensim import corpora
from gensim.test.utils import get_tmpfile

#%%
def json_to_dataframe(input_file_path):
    with open(input_file_path) as json_file:
        json_data = json.load(json_file)
        
    data = json_data['data']
    df = pd.json_normalize(data, ['paragraphs', 'qas', 'answers'], ['title', ['paragraphs', 'context'], ['qas', 'id', 'question']])
    
    return json_data, df

#%%
train_file_path = os.path.join(os.path.dirname(__file__), 'data/train-v2.0.json')
train, df = json_to_dataframe(train_file_path)

#%%
#df['document'] = df['qas.id.question'] + " " + df['title'] + " " + df['paragraphs.context']
df['document'] = df['qas.id.question']
documents = [t for t in df['document'].tolist()]

#%%
def read_corpus(documents, tokens_only=False):
    CUSTOM_FILTERS = [strip_tags, strip_punctuation, remove_stopwords, strip_short, stem_text]
    for i, line in enumerate(documents):
        tokens = preprocessing.preprocess_string(line, CUSTOM_FILTERS)
        if tokens_only:
            yield tokens
        else:
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
        
def tokenize_string(document):
    CUSTOM_FILTERS = [strip_tags, strip_punctuation, remove_stopwords, strip_short, stem_text]
    token = preprocessing.preprocess_string(document, CUSTOM_FILTERS)
    return token

#%%
train_corpus = list(read_corpus(documents))

#%%
#build and training the model
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=20)

model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

#%%
question = 'What branch of theoretical computer science deals with broadly classifying computational problems by difficulty and class of relationship?'
tokenize_question = list(tokenize_string(question))

#%%
vector = model.infer_vector(tokenize_question)

#%%
ranks = []
second_ranks = []

for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    
    second_ranks.append(sims[1])
    
counter = collections.Counter(ranks)


#%%
doc_id = random.randint(0, len(train_corpus) - 1)
# Compare and print the second-most-similar document
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))

#%%
fname = get_tmpfile('chatbot_doc2vec_model')
model.save(fname)

#%%
#Load testdataset
test_file_path = os.path.join(os.path.dirname(__file__), 'data/dev-v2.0.json')
test, df_test = json_to_dataframe(test_file_path)

#%%
test_documents = [t for t in df_test['qas.id.question'].tolist()]
test_corpus = list(read_corpus(test_documents, tokens_only = True))

#%%
#Testing the model
# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test_corpus) - 1)
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
    
print('Question: {}\n'.format(test_documents[doc_id]))
print('True Answer: {}\n'.format(df_test['text'][doc_id]))
sim_id = sims[0]
print('Similar Question: {}\n'.format(df['qas.id.question'][sim_id[0]]))
print('Similar Answer: {}\n'.format(df['text'][sim_id[0]]))
print('Title: {}\n'.format(df['title'][sim_id[0]]))
print('Context: {}\n'.format(df['paragraphs.context'][sim_id[0]]))


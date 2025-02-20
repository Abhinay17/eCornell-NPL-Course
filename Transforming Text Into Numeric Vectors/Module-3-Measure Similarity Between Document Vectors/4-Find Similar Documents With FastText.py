#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a Jupter notebook's cell
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, gensim, nltk, spacy
from collections import Counter
from gensim.models import FastText
from spacy.lang.en import English

np.set_printoptions(linewidth=10000, precision=2, edgeitems=10, suppress=True)
pd.set_option('max_rows', 100, 'max_columns', 1000, 'max_colwidth', 1000, 'precision', 2, 'display.max_rows', 4)
print(f'gensim {gensim.__version__}, numpy {np.__version__}, spacy {spacy.__version__}')

unit = lambda v: v / (v@v)**0.5      # stretch/shrink vector to make it unit length
CS = lambda x, y: unit(x) @ unit(y)  # cosine similarity is a dot product of standardized vectors


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# 
# # **Review**
#  
# In this notebook, you'll train FastText vectors on the MSHA dataset. Then, you'll use similarities calculated from the FastText model to investigate similarity among documents and examine how this metric changes depending on how you train the model.

# ## **Train a FastText Model**

# Use the MSHA dataset to train a FastText model.

# In[2]:


dfAll = pd.read_csv('msha_2003-2018.zip', compression='zip').fillna('')        # replace NaN with empty string
dfAll['INJ_BODY_PART'] = dfAll['INJ_BODY_PART'].apply(lambda s: s.replace('/',', ').replace('(S)','').lower()) # clean up categories
dfAll['NARRATIVE'] = dfAll['NARRATIVE'].str.lower()                            # lower case narratives
dfAll['Len'] = dfAll['NARRATIVE'].apply(len)                                   # narrative's character length
dfDocs = dfAll.sort_values('Len', ascending=False).head(10000)[['NARRATIVE']]  # keep longest 10000 narratives

nlp = English(disable=['tagger', 'parser', 'ner'])  # disabled models save processing time in Spacy
tokenize = lambda text: [t.text.lower() for t in nlp(text)]  # takes a sentences string and produces a list of tokens
dfTok = dfDocs['NARRATIVE'].apply(tokenize).to_frame()
dfTok

get_ipython().run_line_magic('time', 'ft = FastText(sentences=dfTok.NARRATIVE, vector_size=50, workers=16, epochs=10)  # train the model with an list of lists of strings')
# ftmodel.save('msha.bin')                            # save model for later use
# ft = FastText.load('msha.bin')                      # load saved model


# ## **Convert Documents to Vectors**

# Next, you will build vector representations for the documents. Each document has a corresponding vector in a 50-dimensional vector space. You'll just build a few vectors (for short, same-length narratives) for demonstration purposes.

# In[3]:


SSample = dfAll[dfAll.Len==20].NARRATIVE.drop_duplicates().head(20) # pick a few short phrases
SSample.to_frame().T


# Comparing documents across 50 abstract dimensions is difficult, but it can be useful if you proceed with caution. By design, semantically similar documents are close to each other in vector space. While what these 50 dimensions represent is unknown, the direction and magnitude of coefficients can be indicative of document similarity along the given dimension.
#  
# Take a look at the plot of several vectors on several dimensions. The narrative that concerns related topics is likely to have some columns (dimensions) which have the same coloring — say, dark red — to indicate the narratives' similarity to one another along that dimension.

# In[4]:


get_ipython().run_line_magic('time', 'dfSampleVec = pd.DataFrame(list(map(lambda s: ft.wv[s], SSample)), index=SSample)')
dfSampleVec.iloc[:5,:30].style.background_gradient(cmap='coolwarm', axis=1)


# ## **Find Semantically Similar Documents**

# Instead of comparing documents along 50 dimensions, you can summarize the similarity with cosine similarity measure, which is just the cosine of the angle between two 50-dimensional vectors. Display the symmetric matrix of cosine similarities for the vector representations of the narratives.

# In[5]:


from sklearn.metrics.pairwise import cosine_similarity
dfCS = pd.DataFrame(cosine_similarity(dfSampleVec), index=dfSampleVec.index, columns=dfSampleVec.index)
plt.rcParams['figure.figsize'] = [16, 8]   # plot wider figures
ax = sns.heatmap(dfCS, annot=True, cmap='coolwarm', cbar=False);
tmp = ax.set_title('Cosine Similarities Built from FastText Embeddings on Full Sentences');


# The cosine similarities close to 1 (dark red) indicate semantically similar narratives, while dark blue indicate semantically dissimilar narratives. You can evaluate these for yourself by identifying a few dark red and dark blue cells. 
#  
# The diagonal values are not useful because they are always 1, since any sentence or document is perfectly semantically similar to itself.

# ## **Search for Semantic Similarity in a Document**

# Next, you will use a query vector to find narratives, which are semantically related to the corresponding query string. For that, you will need to convert each narrative to a numeric vector and sort the vectors by their cosine similarity to the query vector.
#  
# Below, compute cosine similarity for every narrative in a sample and order these narratives by decreasing cosine similarity so that the most semantically similar narratives are at the top.

# In[6]:


sQuery = 'fell down the stairs'
vQuery = ft.wv[sQuery]
get_ipython().run_line_magic('time', "dfDocs['CS'] = dfDocs.NARRATIVE.head(1000).apply(lambda s: CS(ft.wv[s],vQuery))")
dfDocs.sort_values('CS', ascending=False).head(10)


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Optional Practice**
# 
# Now, equipped with these concepts and tools, you will tackle a few related tasks.
# 
# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.

# # Task 1
#  
# Use the `sQuery = 'loss of hearing'` query string to find semantically related incidents in the `dfDocs.NARRATIVE.head(10000)`. Then, read the "most relevant" narratives and decide on how many are related to the query. What cosine similarity threshold would you use to cut off all unrelated narratives?
# 
#  <b>Hint:</b> This is similar to the search above.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
# <pre>
# sQuery = 'loss of hearing'
# vQuery = ft.wv[sQuery]
# %time dfDocs['CS'] = dfDocs.NARRATIVE.head(10000).apply(lambda s: CS(ft.wv[s],vQuery))
# dfDocs.sort_values('CS', ascending=False).head(10)</pre>
#  
# One challenge with cosine similarity is determining the cut-off threshold. This requires domain expertise, i.e., reading the narratives and making a decision on where the related reports end and unrelated reports begin. Later we will learn additional tools that help us in this unsupervised task.</details>
# </font>

# # Task 2
# 
# Train the FastText model on a larger set of narratives (say, 15,000) in several different ways: for more or fewer epochs and with smaller/larger vector sizes. You can subjectively evaluate the quality of each trained FastText model by observing the relatedness of the results for your `sQuery` above. 
# 
# <b>Hint:</b> This is similar to the training above but with different parameters and longer MSHA text.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
# <pre>
# dfDocs1 = dfAll.sort_values('Len', ascending=False).head(15000)[['NARRATIVE']] # keep longest 15000 narratives
# nlp = English(disable=['tagger', 'parser', 'ner'])           # disabled models save processing time in Spacy
# tokenize = lambda text: [t.text.lower() for t in nlp(text)]  # sentences string -> produces a list of tokens
# dfTok1 = dfDocs1['NARRATIVE'].apply(tokenize).to_frame()
# %time ft = FastText(sentences=dfTok1.NARRATIVE, vector_size=25, workers=16, epochs=20)  # train model</pre>
#  
# This is just one way to retrain the FastText model with 15K sentences and shorter vectors over 20 epochs. Evaluation of the quality of such vectors is needed. One can subjectively assess the model's ability to relate text excerpts. Alternatively, a metric can be used to automate the assessment of such models if many different model variants are tried.</details>
# </font>

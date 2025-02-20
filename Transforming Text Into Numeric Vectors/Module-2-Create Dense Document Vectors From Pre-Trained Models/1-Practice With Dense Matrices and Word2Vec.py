#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.
# 
# Notice that we're upgrading the NumPy and Gensim packages so that they can communicate with each other without errors. Gensim is undergoing rapid development and the package has had several major transformations (hence, version 4). If you experience problems with Gensim in your work, they may be easily fixed by keeping the package up to date.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
# ! pip -q install -U python-Levenshtein==0.12.2 gensim==4.1.0 > log
import pandas as pd, numpy as np, nltk, seaborn as sns, matplotlib.pyplot as plt, gensim
from gensim.models import KeyedVectors

print(f'Versions. gensim:{gensim.__version__}, np:{np.__version__}') 


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # Review
# 
# In this notebook, you will examine a Word2Vec model to practice working with its output and to develop a better understanding of what it can tell you about a particular word. Additionally, you'll develop an appreciation for the limitations of this type of model. 
# 
# ## Word2Vec
# 
# You'll work with the `Gensim`-trained Word2Vec model `'glove-wiki-gigaword-50.gz'`. This is the smallest package in the library and includes about 400,000 words. 
# 
# First, look at each part of the model name, because they give you critical information about the model: 
# 
# 1. [*GloVe*](https://en.wikipedia.org/wiki/GloVe_(machine_learning)) or Global Vectors is the model used to create Word2Vec vectors (aka word-embedding vectors or word embeddings). 
# 1. *wiki* and [*gigaword*](https://catalog.ldc.upenn.edu/LDC2003T05) are large corpora that were used to train the model. Wikipedia 2014 corpus and English Gigaword 5 corpus together had 6 billion of uncased tokens.
# 1. *50* is the size of each vector.
# 1. *.gz* indicates that this is a text file compressed to [*gzip*](https://en.wikipedia.org/wiki/Gzip) format. 
# 
# `'glove-wiki-gigaword-50.gz'` contains a matrix of weights, where each line is a word vector with the word itself starting the line. Use the code below to load this model as `wv`. Note that this may take a minute or two to load.
# 
# Note: The original Word2Vec model and Global Vectors (GloVe) model both produce word vectors, but their algorithms differ in technicalities which we will not discuss here. While you will use the more accessible/popular GloVe embeddings in this course, you will learn about the original Word2Vec algorithm.

# In[2]:


# Dictionary-like object. key=word (string), value=trained embedding coefficients (array of numbers)
sFile = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
get_ipython().run_line_magic('time', 'wv = KeyedVectors.load_word2vec_format(sFile)')
wv            # prints the type of the object and its memory location


# Now that you've loaded the model, retrieve the vector for the word `'cornell'`. 

# In[3]:


wv['cornell']  # retrieve a word vector. Formerly: wv.word_vec('cornell')


# Examine the vector, noting each of the following important characteristics: 
# 
# 1. It contains 50 values somewhere between -2 and +2.
# 1. All values are floats with 32-bit precision.
# 1. There are no zeros, so it can be considered a **dense vector** (a vector comprised of mostly non-zero values).
# 1. Each value represents a dimension but is not necessarily interpretable by humans.
# 1. Large (in magnitude) values, such as `1.73` or `-1.3797`, may relate to education, university, academia, technology, or some other broad category in which Cornell has great presence.
#     1. Many broader categories are also likely to be represented by a combination of these dimensions.
# 1. A few values are relatively close to zero: `0.068273`, `0.045676`, `0.034797`, `-0.074928`.
#     1. This implies that these dimensions do not carry "significant information" about `'cornell'`.
# 
# Next, check the vector length with the model's `vector_size` attribute by taking the length of any vector you obtain from the model. Note, however, that words are only present if they were present in the training corpus. The word `'the'` is highly likely to be present in all models, so it is a good choice if you want to check the number of dimensions.

# In[4]:


wv.vector_size
len(wv['the'])


# If a word in which you're interested is not in a model, some preprocessing may help. In the previous example, you used `'cornell'` instead of `'Cornell'` since the uppercase version `'Cornell'` is not in this model's vocabulary. 
# 
# To check if a word is included the vocabulary, you can use the `key_to_index` attribute. 

# In[5]:


'Cornell' in wv.key_to_index, 'cornell' in wv.key_to_index


# You can check the size of a particular model with the `len` method. 400K words may seem like a lot, but this is actually quite small for most real-world applications. In practice, you can use models with larger vector sizes (say, 300) and much larger vocabulary.

# In[6]:


len(wv.key_to_index)  # retrieve vocabulary and measure its size


# Let's print a few words from this model's vocabulary. Since `wv.key_to_index` is a dictionary, you need to wrap it as a list before slicing.

# In[7]:


LsTop20 = list(wv.key_to_index)[:20]
print(LsTop20)


# You can go a step further and print vectors associated with each of these words. Use the `background_gradient` method to wrap these vectors into a DataFrame as rows with the corresponding words as row indices. Let's spice it up with colors. WOW! 

# In[8]:


pd.DataFrame({w:wv[w] for w in LsTop20}).T.style.background_gradient(cmap='coolwarm').set_precision(2)


# Note that the 50 columns you see here are dimensions you examined earlier. We do not know what each dimension signifies, although this is an active area of research. However, if you examine each dimension, you'll notice more red in some columns and more blue in others. The words we loaded above are very common and  might be stopwords we would normally remove. However, if we load more specific groups of words, we might make good guesses about what some of the dimensions mean.
# 
# Now that you've explored the model programmatically, consider downloading this model, unzipping it, then opening it in the text editor and searching for the example words we reviewed above. Explore the vectors and confirm that they are the same as what you retrieve programmatically.

# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Optional Practice**
# 
# Now, equipped with these concepts and tools, you will practice a few related tasks.
# 
# 
# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.

# ## Task 1
# 
# First, evaluate the similarity between two word vectors the word vectors for `'university'` and `'cornell'` simply by counting the matched signs of their coefficients (report as a fraction of matched signs). Try comparing other words to these two to find some that are more or less similar.
# 
# <b>Hint:</b> You can create a mask from comparing each vector as > 0. Then compare two masks and count the matches. Then divide by 50.
# 

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# # Solution 1
# matched_signs = sum((wv['university'] > 0) == (wv['cornell'] > 0)) 
# matched_signs / 50
# # Solution 2
# np.mean((wv['university'] * wv['cornell']) > 0)
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# Now, for the top 500 words in `wv`'s vocabulary, find the top three that are most similar to `'cornell'` using the metric we created in the previous task. Do the results make sense?
# 
# <b>Hint:</b> Try wrapping vocabulary dictionary as a list then retrieving 500 words and for each one compute the metric above. If you wrap it as a list of tuples with the first element as the measure and the second as the word, then you can easily apply <code>sorted()</code> to order these tuples by similarity score. You can retrieve the bottom three tuples.
# 

# In[ ]:


# check solution here


# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# sorted((sum((wv[w] > 0) == (wv['cornell'] > 0)) / 50, w) for w in list(wv.key_to_index)[:500])[-3:]
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# Similarly, for the top 500 words in `wv`'s vocabulary, find the top three that are least similar to `'cornell'` using the metric you created in the previous task. Do the results make sense?
# 
# <b>Hint:</b> This is similar to the code above but is retrieving elements from the other end of the list.
# 

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# sorted((sum((wv[w] > 0) == (wv['cornell'] > 0)) / 50, w) for w in list(wv.key_to_index)[:500])[:3]
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 4
# 
# Retrieve all words from `wv` that contain the word `'university'`. What separators do you observe? Any punctuation or uppercasing?
# 
# <b>Hint:</b> Try it with list comprehension.
# 

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# print([w for w in wv.key_to_index if 'university' in w])
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 5
# 
# Retrieve all words from `wv` that contain the words `'new'` and `'york'`. What separators do you observe?
# 
# <b>Hint:</b> This is similar to the code above, but you need two comparisons in your condition statement.
# 

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# [w for w in wv.key_to_index if 'new' in w and 'york' in w]
#     </pre>
#     </details> 
# </font>
# <hr>

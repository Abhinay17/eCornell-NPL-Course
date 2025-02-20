#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.
# 
# <span style="color:black">You will use `nltk`'s [`PorterStemmer()`](https://www.nltk.org/howto/stem.html) and [`WordNetLemmatizer()`](https://www.nltk.org/api/nltk.stem.html#nltk.stem.wordnet.WordNetLemmatizer) methods to stem and lemmatize the Brown corpus, respectively.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
import nltk, pandas as pd
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ
tmp = nltk.download(['brown','wordnet'], quiet=True)
pso = nltk.stem.PorterStemmer()       # Porter stemmer object
wlo = nltk.stem.WordNetLemmatizer()   # WordNet lemmatizer object


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # Review
# 
# 
# `nltk` offers several stemming and lemmatization libraries, but the two methods you'll practice below are most popular and are reasonably effective. 
# 
# ## Stem and Lemmatize Words
# 
# The `PorterStemmer()` method operates on the given word alone. 
# 
# The `WordNetLemmatizer()` method also expects a part of speech (POS) tag for the given word. This means that you must indicate whether the word is a verb (tag `'v'`) or a noun (`'n'`) or an adjective (`'a'`) or an adverb (`'r'`). The default is `'n'`, which means all words are treated as nouns. Unfortunately, this means that verbs that don't have a noun form in the WordNet database are left unmodified. To raise the quality of the lemmatizer, you should provide a word with its POS tag. In the next module you will discover how to automate this task. For now, you will create three lemmatizing wrappers with hardcoded POS tags: `LemN`, `LemA`, and `LemV`.
# 

# In[2]:


Stem = lambda s: pso.stem(s)          # lambda function is a simplified function
LemN = lambda s: wlo.lemmatize(s, NOUN)
LemA = lambda s: wlo.lemmatize(s, ADJ)
LemV = lambda s: wlo.lemmatize(s, VERB)


# Apply the stemmer and three lemmatizers to each word in a `LsWords` to evaluate the standardization effect.

# In[3]:


LsWords = ['running','corpora','drove','tries','asked','agreed','oldest','incubation', 'debug']
LTsStd = [(s, Stem(s), LemN(s), LemA(s), LemV(s)) for s in LsWords]
LTsStd


# Wrap these results into a neat table using Pandas `DataFrame` object, which has a myriad of convenient attributes and methods for sorting, filtering, and otherwise manipulating the table.

# In[4]:


df = pd.DataFrame(LTsStd, columns=['Orig','Stem','Lemma|Noun','Lemma|Adj','Lemma|Verb'])
df


# ## Find Incorrect Words
# 
# Our goal is to find stemmed and lemmatized words that are incorrect. One way of doing so is to check the augmented words against some large lexicon, which contains most common words. The [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus) is often used for this purpose. It contains over a million words, 56,000 of which are unique. So, we remove all duplicates by applying the `set()` function on the list of returned words.
# 

# In[5]:


LsBrownWords = nltk.corpus.brown.words()  # list of all word tokens
SsBrownWords = set(LsBrownWords)          # set of unique words
print(f'Tokens:{len(LsBrownWords):,}; Unique words:{len(SsBrownWords):,};', LsBrownWords[:20])


# Now, reformat the stemmed and lemmatized words. The `values` attribute of a dataframe creates a NumPy array of elements.

# In[6]:


df.values


# Flatten this list of lists using NumPy's [`flatten()`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html) or [`ravel()`](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html) method. The second is actually preferred, since it tries to avoid copying data in memory, if possible.

# In[7]:


SsStd = set(df.values.flatten())
print(SsStd)


# Finally, you can test whether these stemmed or lemmatized words are in the set of Brown words.

# In[8]:


[s for s in SsStd if s not in SsBrownWords] # find incorrectly standardized words


# <span style="color:black">While it is reasonable to consider the words that are not present in this set as incorrectly spelled words, there are a few problems. `'debug'` and `'corpora'` are real words, but they are not in the Brown Corpus. If you want to improve the English language lexicon that is used for this comparison, you can add words from other corpora and add lowercased Brown words.

# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Optional Practice**
# 
# You will now practice stemming and lemmatizing.
#     
# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you've gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.

# ## Task 1
# 
# Use `nltk.download()` to load the `'brown'` Corpus to local storage. Then, use `nltk.corpus.brown.words()` to load `'cj06'`, a list of words, to some variable in memory. Keep only words of length 3 or more and that contain only letters. Remove duplicates by converting this list to a set named `Ss6`. You should end up with 562 unique words with three or more letters.
# 
# <b>Hint:</b> See some examples of loading full Brown <a href="https://www.nltk.org/book/ch02.html">here</a>.
# 

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# _ = nltk.download(['brown'], quiet=True)
# Ss6 = {s for s in nltk.corpus.brown.words('cj06') if s.isalpha() and len(s)>2}
# print(len(Ss6))
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# Since all these words came from the document `cj06` that you already loaded to `SsBrownWords`, the set `Ss6` should have no words outside of the Brown Corpus. Verify this.
# 
# <b>Hint:</b> You can try set difference (which is faster) or list/set comprehension. Either should return a blank container indicating that no words of <code>Ss6</code> are found in <code>SsBrownWords</code> lexicon.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# # Solution 1: set difference: elements of Ss6, which are not in SsBrownWords
# Ss6 - SsBrownWords   
# # Solution 2: list comprehension 
# [s for s in Ss6 if s not in SsBrownWords]
# # Solution 3: returns True if all Ss6 words in SsBrownWords
# Ss6.issubset(SsBrownWords) 
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# Now, apply the `Stem()` function to the words in `Ss6` and count the number of **new** words that are not in `Ss6`.
# 
# <b>Hint:</b> You can use set comprehension to iterate and stem each word in <code>Ss6</code>. Then use set difference to subtract elements of <code>Ss6</code>.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# SsStem6 = {Stem(s) for s in Ss6} - Ss6  # stems which are not in Ss6
# print(len(SsStem6), sorted(SsStem6)[:20])
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 4
# 
# Among these new words that result from stemming, find those that are not in your Brown Corpus `SsBrownWords`. There should be 204 such words, which is almost half of the number of words that were originally in `Ss6`. 
# 
# Note that, while trying to standardize the document vocabulary, you created many words that are not in the English vocabulary.
# 
# <b>Hint:</b> Try set difference between <code>SsStem6</code> and <code>SsBrownWords</code>.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# SsStem6_bad = SsStem6 - SsBrownWords # stems which are not words
# print(len(SsStem6_bad), sorted(SsStem6_bad)[:20])
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 5
# 
# Now, let's evaluate how many out-of-vocabulary words you derive from lemmatization using the default part-of-speech (POS), i.e., noun tag. Similar to above, create a set variable `SsLem6`, which contains all string words after applying `LemN()` and removing original words from `Ss6` set. Save the result to `SsLem6`.
# 
# There should be 32 such words, which is only a fraction compared to those generated from stemming above. Moreover, note that many of these words are sensible English words. Can you determine and/or investigate why these words were not in `Ss6` originally?
# 
# <b>Hint:</b> Try set difference as you did above. One hypothesis is that these words were in `Ss6` in their plural form, but not in singular form. Try searching for others.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# Lem = LemN
# SsLem6 = {Lem(s) for s in Ss6} - Ss6  # lemmas which are not in Ss6
# print(len(SsLem6), sorted(SsLem6)[:20])
# 'chips' in Ss6, 'chip' in Ss6
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 6
# 
# 
# Finally, find all words in the set  `SsLem6` that are not in the Brown Corpus. There should be five such words, some of which are highly technical terms, and which are therefore rare in general text.
# 
# <b>Hint:</b> Try set difference as you did above.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# SsLem6_bad = SsLem6 - SsBrownWords # lemmas which are not words
# print(len(SsLem6_bad), sorted(SsLem6_bad)[:20])
#             </pre>
#     </details> 
# </font>
# <hr>

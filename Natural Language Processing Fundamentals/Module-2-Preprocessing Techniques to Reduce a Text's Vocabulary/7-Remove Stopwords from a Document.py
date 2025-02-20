#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.
# 
# <span style="color:black">You will practice removing the stopwords in "Persuasion," a novel by Jane Austen. Use `nltk` to load this text from the Gutenberg (free) library.

# In[2]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import nltk, pandas as pd
from collections import Counter
tmp = nltk.download(['gutenberg','stopwords'], quiet=True)

sTitle = 'austen-persuasion.txt'
print(nltk.corpus.gutenberg.raw(sTitle)[:200]) # print the top few characters


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# ## Examining English Stopwords
# 
# <span style="color:black"> `nltk` provides lists of generic stopwords for different languages. Load the stopwords for the English language as a set. Take time to examine the length of this set and the sorted list of words.

# In[3]:


SsStopwords = set(nltk.corpus.stopwords.words('english')) # load generic stopwords
print(f'stopwords:{len(SsStopwords)}')
print(sorted(SsStopwords))


# <span style="color:black"> Notice that the stopwords are lowercase. This is important; when you remove stopwords from a text, you will need to make sure that you are comparing the lowercase version of the words in the text to the stopwords in this list.
#     
# <span style="color:black">The excellent people who built `nltk` identified 179 generic English stopwords. You may disagree with some of their choices, and rightly so. A different list may be better suited for a corpus in a specific domain. You can manually add more words to this list or implement an automated method to find the stopwords in your document with [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) concepts that will be discussed in a later course.

# ## Calculating Average Word Frequency
# 
# For some NLP tasks, you may want to reduce the number of unique words in a document while minimizing its effect on the semantics of the document. _Average frequency_, i.e., the average number of times a word is repeated in the document, can be a useful metric for determining which words to remove.
#     
# Load the novel as a list of words and calculate the number of word tokens, number of unique words, and average frequency.

# In[5]:


def DocStats(Ls=[]): 
  nL, nS = len(Ls), len(set(Ls))
  print(f'Tokens: {nL}; Lexicon: {nS}; Avg Freq: {nL/nS:.3f}')

LsBook = nltk.corpus.gutenberg.words(sTitle)
DocStats(LsBook)


# The distribution of word frequency in a document is typically not uniform. For example, it is common for stopwords to appear at a significantly higher frequency. Remove the `nltk` stopwords from the novel then recompute the statistics.

# In[6]:


LsBook2 = [w for w in LsBook if w.lower() not in SsStopwords]
DocStats(LsBook2)


# By comparing `LsBook` and `LsBook2`, you can see that removing the stopwords significantly compressed the corpus, reducing the number of words by approximately half. As expected, the average frequency of the remaining words is lower because stopwords often appear at higher frequencies.
# 
# ## Examining the Impact of Stopword Removal
# 
# To determine the impact of the removed words, you can compare the top n-most frequent words in the previous two examples. Use the `most_common()` method from the `Counter` object to count the parsed words from the novel and get the top 100 most frequent words. Package the results as a Pandas dataframe and order the words by decreasing frequency.

# In[7]:


LTsnFreq = Counter(LsBook).most_common(100) 
pd.DataFrame(LTsnFreq, columns=['word','freq']).set_index('word').T


# <span style="color:black">Notice the large count of several stopwords. 'a' appears 1529 times in the novel, yet, if you removed this word, you can still read and understand the novel. On the other hand, if you removed the word 'Russell', which appears only 148 times, you would find it difficult to recover the context and potentially incorrectly associate descriptions in the novel to another character.
# 
# <span style="color:black">Now, examine the 100 most frequent words without the generic stopwords from `nltk`.

# In[8]:


pd.DataFrame(Counter(LsBook2).most_common(100), columns=['word','freq']).set_index('word').T


# <span style="color:black">In this example, most generic words were removed by just using the `nltk` list. If punctuation is not important in your NLP task, you can also include these in the stopword list.
# 
# <span style="color:black"> If you want to further reduce the document by another thousand tokens, you can remove several of the high-frequency words, including 'could' and 'would', without significantly affecting semantics. In fact, high-frequency words can be considered stopwords. However, as words are removed, distinguishing which to remove becomes harder since frequency starts to drop and words become more important to the document. You will learn about a better method, TF-IDF, in the future.

# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Optional Practice**
# 
# 
# You will now practice removing stopwords.
# 
# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you've gotten the correct result. If you get stuck on a task, click the See **solution** drop-down menu to view the answer.

# # Task 1
# 
# You can further clean the novel in `LsBook2` by keeping *letter* words with at least 3 characters. Save results to `LsBook3` and run `DocStats()` on it to evaluate the decrease in counts and average frequency.
#  
# Average frequency should drop by about 2 points to 6.531. This is still a significant drop, but it will be harder and harder to identify and keep "high quality" words, which an average reader would associate with this novel.
# 
# <b>Hint:</b> You can do this with list comprehension (or any loop) and condition <code>len(w)>2 and w.isalpha()</code>, where <code>w</code> is a word from <code>LsBook2</code>.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# LsBook3 = [w for w in LsBook2 if len(w)>2 and w.isalpha()]
# DocStats(LsBook3)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# Use `Counter()` to compute the top 100 most common word counts in `LsBook3` and save these to `LTsnTop3`. Wrap them into a dataframe for a nice horizontal display. 
# 
# <b>Hint:</b> Check out examples of counting with the <code>Counter</code> object from <a href="https://docs.python.org/3/library/collections.html#collections.Counter">the Python documentation</a>. This is similar to what you did above and in the video.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# LTsnTop3 = Counter(LsBook3).most_common(100)
# df = pd.DataFrame(LTsnTop3, columns=['word','freq']).set_index('word')
# df.T
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# Retrieve only the words from `LTsnTop3` and save the set of these to `SsTop3`.
# 
# <b>Hint:</b> You can use <code>zip</code> function to rearrange elements of list of tuples returned from <code>Counter().most_common()</code> or use <code>df.index</code> to access index labels, which are the same words.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# SsTop3 = set(list(zip(*LTsnTop3))[0])
# # SsTop3 = list(df.index)   # alternative extraction of words from df's index
# print(len(SsTop3), sorted(SsTop3))
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 4
# 
# Retrieve all words from `SsTop3` that are title cased and save the set of these to `SsTopTitleCase`.
# 
# Note: Many of these words are peoples' names, but not all. There are also military ranks and common salutations.
# 
# <b>Hint:</b> You can use set comprehension with a condition on to check whether a word string is title cased. Consider the <a href="https://docs.python.org/3/library/stdtypes.html#str.istitle"><code>str.istitle()</code></a> method.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# SsTopTitleCase = {w for w in SsTop3 if w.istitle()}
# print(len(SsTopTitleCase), sorted(SsTopTitleCase))
#             </pre>
#     </details> 
# </font>
# <hr>
# 

# ## Task 5
# 
# Observe the words in `SsTopTitleCase` and manually identify those that you consider to be generic. Remove these words from `SsTopTitleCase` and save the results to `SsTopNames`.
# 
# <b>Hint:</b> try <a href="https://docs.python.org/3/tutorial/datastructures.html#sets">set differencing</a>.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# SsOtherGeneric = {'Sir', 'Bath', 'Miss', 'Mrs', 'Lady', 'Uppercross'}
# SsTopNames = SsTopTitleCase - SsOtherGeneric # set difference
# # SsTopNames = {w for w in SsTopTitleCase if w not in SsStopwords} # alternative approach
# print(list(SsTopNames))
#             </pre>
#     </details> 
# </font>
# 
# <hr>

# ## Task 6
# 
# Remove people's names stored in `SsTopNames` from the top 100 most frequent words stored in `SsTop3`. Save results to `SsTopNoNames`, which should contain about 80 high frequency words (depending on which ones you picked above).
# 
# <b>Hint:</b> Try <a href="https://docs.python.org/3/tutorial/datastructures.html#sets">set differencing</a>.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# SsTopNoNames = SsTop3 - SsTopNames
# print(len(SsTopNoNames), SsTopNoNames)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 7
# 
# Before you merge `SsTopNoNames` with the `SsStopwords`, look at it once more. Remove any words that you don't consider to be stopwords (in your opinion). Notably, this step requires reasonable domain expertise, i.e., understanding the value of these words in Jane Austen's novel. A greater expertise is required if you are to continue identifying low-value words.
#  
# Finally, combine the two sets and save results to `SsStopwordsXtra`, which should now contain about 260 stopwords.
# 
# <b>Hint:</b> try <a href="https://docs.python.org/3/tutorial/datastructures.html#sets">set union</a>.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# SsStopwordsXtra = SsStopwords.union(SsTopNoNames)
# len(SsStopwordsXtra)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 8
# 
# Create `LsBook4`, which contains all the words in `LsBook3` excluding those in `SsStopwordsXtra` (case insensitive). Apply `DocStats()` to `LsBook4`. The average frequency should drop to abot 5.2. Congratulations!
# 
# <b>Hint:</b> This is similar to the code above where we created <code>LsBook2</code>.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# LsBook4 = [w for w in LsBook3 if w.lower() not in SsStopwordsXtra]
# DocStats(LsBook4)
# print(LsBook4[:20])
#             </pre>
#     </details> 
# </font>
# <hr>

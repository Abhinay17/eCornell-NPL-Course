#!/usr/bin/env python
# coding: utf-8

# # **Setup**
# 
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.   

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import numpy as np, pandas as pd, nltk, re, pprint


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# You will use the `nltk` package, which is the Natural Language Toolkit, to implement the tokenization methods  Professor Melnikov discussed in the previous videos. This Python toolkit is extremely useful for NLP tasks because it includes common string processing methods and sample corpora from the Project Gutenberg archives. 
#     
# Load the novel, "Alice's Adventures in Wonderland" by Lewis Carol, and print the first 200 characters.

# In[2]:


_ = nltk.download(['punkt', 'gutenberg'], quiet=True)
sDoc = nltk.corpus.gutenberg.raw('carroll-alice.txt')
print(sDoc[:200])


# ## Tokenizing (Parsing) Sentences
# 
# 
# The [`nltk.sent_tokenize(text, language='english')`](https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.sent_tokenize) method accepts two arguments, the text that you want to parse into sentences and the language of the text, and returns the list of sentence tokens. If the language is not specified, the method uses English as the default langauge. Below it is used to tokenize the novel's text and print out the first five sentence tokens. Note that splitting is done not just on spaces, but on characters and combinations of characters indicating end of sentences (such as '. ', '! ', etc.)
# 
# Recall from your (prerequisite) Python course that a [lambda](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) function is a **nameless** function, which we sometimes give a name (with a variable assignment notation). It has a simpler and more compact structure, but is limited in the number of operations outside of function compositions.

# In[3]:


# The function IndexSents iterates through a list of tokens and add indices to the beginning of the tokens
IndexSents = lambda LsSents: [f' <<{i}>> {s}' for i, s in enumerate(LsSents)]
IndexSents(nltk.sent_tokenize(text=sDoc)[:5])


# If you are uncomfortable or unfamiliar with [list comprehensions](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions), the following example would be another way to code using a traditional function definition with a standard for loop. You are encouraged to copy this cell into a new cell block and uncomment the print statements if you would like to see how each step of the process is working.

# In[4]:


def SimpleIndSent(tok_list):
    LsSents = []
    for i in range(0,len(tok_list)):
        #print(i)
        #print(tok_list[i])
        number = "<<" + str(i) + ">>"
        #print(number)
        LsSents.append(number + tok_list[i])
        #print(LsSents[i])
    return LsSents


# In[5]:


# Compare the outputs of the two functions to confirm they are identical.
SimpleIndSent(nltk.sent_tokenize(text=sDoc)[:5])


# Observe the output. Some of the sentence tokens above combine multiple independent phrases that should have been tokenized into individual sentences. One way to improve sentence tokenization is to replace newline (`\n`) characters with punctuation. For example: 
# * two or more newline characters, `'\n\n'`, are used to separate paragraphs. You can replace them with a sentence separator, `'. '`. 
# * a single newline character, `'\n'`, is typically used as a line wrapper. You can replace it with a space. 
# 
# Use these examples to replace newline characters in the novel, then tokenize the processed text.

# Step 1: Replace two or more adjacent newlines with a '. ', a sentence separator.

# In[6]:


sDoc1 = re.sub('\n\n+', '. ', sDoc)  
sDoc1


# Step 2: Replace a single newline with a space.

# In[7]:


sDoc1 = re.sub('\n', ' ', sDoc1)      
sDoc1


# Step 3: Create a list of sentence tokens. 

# In[8]:


LsSent1 = nltk.sent_tokenize(sDoc1)
LsSent1


# Now that you've replaced the newline characters, notice that the sentence tokens below make more sense.

# In[9]:


IndexSents(LsSent1[:7])


# ### Ensuring Sentences are Parsed Correctly at Scale
# 
# How can you ensure that the other sentences in the novel are parsed correctly? Spotting anomalies is easy when examining only the first few sentence tokens, but if you were to do this for the rest of the novel, you might as well manually parse the sentences. Remember, our goal is to automate text processing.
#     
# One approach is to investigate the extreme cases, such as the longest sentences. The [`sorted()`](https://docs.python.org/3/library/functions.html#sorted) method is valuable for finding extreme values. One way you can use `sorted()` is by building tuples of sentences and their lengths with list comprehension, then sorting it by the second element (i.e. the length element of the tuple) in reverse (i.e. descending) order.
#    

# In[10]:


IndexSents(sorted([(s, len(s)) for s in LsSent1], key=lambda s_len: s_len[1], reverse=True)[:2])


# The top two longest sentence tokens, displayed here in the output, have 1,191 and 1,155 characters, respectively. These tokens are very long, which is unlikely in a children's book, so they are probably incorrect. Observe that both sentence tokens contain multiple sentences separated by `'.. '`, which were undetected by the parser. To improve parsing on this document, you can choose between the following two methods, depending on your expertise with regex, your comfort level modifying the underlying NLTK code, and your desired processing speed.
# 
# 1. Replace contiguous periods with a single period, assuming that double and triple periods are only used for sentence separation by the author. This hypothesis can (and should be) be checked, of course.
# 1. Modify the regex string in the underlying [`PunktSentenceTokenizer()`](https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.punkt.PunktSentenceTokenizer) parser to account for adjacent periods and even newlines. This method is faster, since regex is typically applied in a single pass, but requires a bit more implementation and validation on your end. This method goes beyond the scope of this class.
#    
# Мore examples with `sorted()` can be found [here](https://docs.python.org/3/howto/sorting.html).
# 
# Now, evaluate the shortest sentences.

# In[11]:


IndexSents(sorted([(s, len(s)) for s in LsSent1], key=lambda s_len: s_len[1], reverse=False)[:6])


# These tokens appear to be correctly parsed since they all end with punctuation.

# ## Tokenizing Words
# 
# The `nltk` library also includes a word tokenization method, [`nltk.word_tokenize(text, language='english', preserve_line=False)`](https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.word_tokenize). Apply this method to the novel text and print out the first 100 word tokens.

# In[12]:


LsWords = nltk.word_tokenize(sDoc)   # list of word strings
print(LsWords[:100])


# Note that this word tokenizer considers punctuations, brackets, quotes, numbers, etc. as words tokens. If punctuation is not important to your task, you can remove them after parsing (or in theory, even during parsing by tuning the underlying [`PunktSentenceTokenizer()`](https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.punkt.PunktSentenceTokenizer) class. However, as previously mentioned, modifying default NLTK behavior is beyond the scope of this class.) Alternatively, you can also use different tokenizers, such as [spaCy's tokenizers](https://spacy.io/api/tokenizer). These tokenizers offer high quality parsing for sentences and words, but are slower than regex-based tokenizers.

# In[13]:


LTsnWords = [(s, len(s)) for s in LsWords]
print(sorted(LTsnWords, key=lambda s_len: s_len[1])[-20:])


# If we look at the longest words, we see several dash-separated phrases. It is up to us to decide whether these should be considered as words or parsed further. Evaluating the context of these phrases will be helpful. Notably, there are no extremely long words in this set of tokens
# 

# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # Optional Practice
# 
# You will now practice more preprocessing tasks using the variables you created above.
# 
# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you've achieved the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer. 

# ## Task 1
# 
# Use `LTsnWords` and return a list of top twenty tuples (word token, length) ordered by increasing word length.
# 
# <b>Hint:</b> Try the built-in <a src="https://docs.python.org/3/library/functions.html#sorted"><code>sorted(iterable, key=None)</code></a> function with a key argument, which sorts by the length (i.e., second) element of the tuple element of the list. Then you can slice the first 20 elements of the resulting list. There are other ways to achieve the same result. For example, you can do all sorting with Pandas DataFrames.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# print(sorted(LTsnWords, key=lambda s_len: s_len[1])[:20])
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# Clean up `LsWords` by removing all duplicates and single character tokens that are punctuation, e.g., remove `';'`, but not `"'the"`. Use the variable [`punctuation`](https://docs.python.org/3/library/string.html#string.punctuation) from the [`string`](https://docs.python.org/3/library/string.html) library that contains a string of common punctuation symbols. Save the results as a list of unique words, `LsVocab`, and calculate the number of words in this list.
# 
# <b>Hint:</b> Try <a href="https://docs.python.org/3/tutorial/datastructures.html#sets">set comprehension</a> to drop duplicates and to condition on non-punctuation tokens. You can apply <code>list()</code> to the output set.

# In[ ]:


from string import punctuation
# check solution here


# 
# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>          
# # Solution 1:
# LsVocab = [s for s in set(LsWords) if not s in punctuation]  # list of unique tokens without punctuation tokens
# len(LsVocab)
# 
# \# Solution 2:
# LsVocab = list({s for s in LsWords if not s in punctuation})  # list of unique tokens without punctuation tokens
# len(LsVocab)
# 
# \# Solution 3: This solution is more efficient in most practical cases (i.e., ones with many duplicates)
# import string
# LsVocab = set(LsWords) - set(string.punctuation)
# len(LsVocab)
# 
# \# Solution 4:
# index = np.isin(LsWords, list(punctuation), invert=True)
# len(np.ndarray.tolist(np.unique(np.asarray(LsWords)[index])))
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# Get the list of word tokens which are not just letters and not just digits from `LsVocab`. Then save it as `LsVocabPunkt`. Calculate the number of words in this list. Note that there is a string method that can check whether all characters in a string are alphabet letters.
# 
# Correct output will contain at least one letter and at least one symbol, as in the examples below:
# - `sky-rocket`
# - `'Why`
# - `they're`
# 
# Note: In evaluating the quality of your word tokenizer, you might want to focus your attention on the shorter list of tokens. If the tokens appear unusual, you should  investigate their context in the original document to decide how to preprocess the document to improve tokenization. For example, you might observe many contraction parts or words. Later you will learn contraction expansion techniques that can help you with such cleaning.
# 
# <b>Hint:</b> Try a <a href="https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions">conditional list comprehension</a> with <code>str.isalpha()</code> and <code>str.isdigit()</code> filters. The first five elements in the resulting <code>LsVocabPunkt</code> list should be: 
# <pre>["'Pepper", "'Why", "'Tell", "'Drink", 'pig-baby']</pre>

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>
#             See <b>solution</b>.
#         </summary>
#            <pre>
# # Solution 1:
# LsVocabPunkt = [s for s in LsVocab if not s.isalpha() and not s.isdigit()] 
# len(LsVocabPunkt)
# 
# \# Solution 2:
# LsVocabPunkt = [s for s in LsVocab if not s.isalnum()] 
# len(LsVocabPunkt)
#             </pre>
#     </details> 
# </font>
# <hr>

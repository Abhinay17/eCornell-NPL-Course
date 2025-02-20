#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
get_ipython().system('pip -q install contractions > tmp   # install contractions package')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import nltk, re, pandas as pd, contractions
_ = nltk.download(['gutenberg'], quiet=True)
sAlice = nltk.corpus.gutenberg.raw(fileids='carroll-alice.txt').lower()


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# Contractions make speaking a language easier, but they increase the size of the vocabulary in your document without adding any information, so it is best to expand them into their uncontracted form as a part of preprocessing. Here, you'll review three types of rules you can use to expand contractions. 
# 
# ## Generic Rules
# 
# 
# A generic contraction expansion rule focuses on the [clitic](https://en.wikipedia.org/wiki/Clitic), i.e., the characters representing the second word in the contraction. For example, the `'m` postfix in `I'm` is a clitic. The advantage is that with a few generic rules, we can expand most contractions. Here, you'll create a function that applies regex substitution to several clitics. 

# In[2]:


def unContract_generic(sTxt='') -> str:
    '''Search and replace generic contraction forms.
    Input: 
        sTxt: input string with contractions
    Returns:
        string with expanded contractions   '''
    # substitute pattern of a string sTxt with expansion replacement
    sTxt = re.sub(r"n\'t", " not", sTxt)
    sTxt = re.sub(r"\'re", " are", sTxt)
    sTxt = re.sub(r"\'s", " is", sTxt)
    sTxt = re.sub(r"\'d", " would", sTxt)
    sTxt = re.sub(r"\'ll", " will", sTxt)
    sTxt = re.sub(r"\'t", " not", sTxt)
    sTxt = re.sub(r"\'ve", " have", sTxt)
    sTxt = re.sub(r"\'m", " am", sTxt)
    return sTxt


# Apply this function to a simple phrase.

# In[3]:


sTxt = "Now's the time when NLP's booming."
print(unContract_generic(sTxt))


# The function worked well in this example. Most contractions can be expanded using a few generic rules, but it can be risky to use these rules broadly. A clitic `'s` can be expanded into multiple variants, such as `was`, `is`, `has`. It can also mean a plural form for some words, such as `A's and B's`, and a [possessive](https://en.wikipedia.org/wiki/Possessive) form of a word, such as `no man's land`.
# 

# In[4]:


sTxt = "You're welcome in Ed's kitchen"
print(unContract_generic(sTxt))


# <span style="color:black"> The function incorrectly expanded `Ed's` kitchen, because the  `'s` is a possessive.

# ## Specific Rules
# 
# To lessen the risk of incorrectly expanding contractions with a generic rule, you can develop a specific set of rules for contraction expansion. To search a string for all contractions in a single pass, you can first package them in a dictionary as key-value pairs, where the key is contraction and its value is the expansion. Then, you can use regex to compile a single search string of all keys. This approach is more computationally efficient than a multi-pass search through the string, especially for very large corpora. 
#     
# `ContractionsMap` is a dictionary with several specific rules. Note that this set of contractions is still too small to express all possible contractions, since there are at least as many of them as there are nouns in the English language. However, this set covers most commonly observed cases. 
# 

# In[5]:


ContractionsMap = { 
    "ain't": "am not", # / are not / is not / has not / have not",
    "aren't": "are not", # / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had", # / he would",
    "he'd've": "he would have",
    "he'll": "he shall", # / he will",
    "he'll've": "he shall have", # / he will have",
    "he's": "he has", # / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has", # / how is / how does",
    "I'd": "I had", # / I would",
    "I'd've": "I would have",
    "I'll": "I shall", # / I will",
    "I'll've": "I shall have", # / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had", # / it would",
    "it'd've": "it would have",
    "it'll": "it shall", # / it will",
    "it'll've": "it shall have", # / it will have",
    "it's": "it has", # / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had", # / she would",
    "she'd've": "she would have",
    "she'll": "she shall", # / she will",
    "she'll've": "she shall have:, # / she will have",
    "she's": "she has", # / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as", # / so is",
    "that'd": "that would", # / that had",
    "that'd've": "that would have",
    "that's": "that has", # / that is",
    "there'd": "there had", # / there would",
    "there'd've": "there would have",
    "there's": "there has", # / there is",
    "they'd": "they had", # / they would",
    "they'd've": "they would have",
    "they'll": "they shall", # / they will",
    "they'll've": "they shall have", # / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had", # / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall", # / what will",
    "what'll've": "what shall have", # / what will have",
    "what're": "what are",
    "what's": "what has", # / what is",
    "what've": "what have",
    "when's": "when has", # / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has", # / where is",
    "where've": "where have",
    "who'll": "who shall", # / who will",
    "who'll've": "who shall have", # / who will have",
    "who's": "who has", # / who is",
    "who've": "who have",
    "why's": "why has", # / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had", # / you would",
    "you'd've": "you would have",
    "you'll": "you shall", # / you will",
    "you'll've": "you shall have", # / you will have",
    "you're": "you are",
    "you've": "you have"
}


# Compile a regex search of contraction keys from `ContractionsMap` and replace the match with the corresponding expansion. The trick is to combine all keys containing contractions into a single regex pattern string.
# 

# In[6]:


'(%s)' % '|'.join(ContractionsMap.keys())   # combine all dictionary keys containing contraction words


# Carefully walk-through the `unContract_specific()` function to understand what it does to keys and values of the dictionary `ContractionsMap`. The `re.compile()` precompiles a regex string to speed up the regex search  further. A helper function `ReplaceMatches()` is passed to the regex's `sub()` method. Whenever it finds a matching contraction, it returns the corresponding expansion for the matched string of [`re.Match`](https://docs.python.org/3/library/re.html#match-objects) object.

# In[7]:


def unContract_specific(sTxt='', cmap=ContractionsMap) -> str:
    '''Expand contractions in sTxt string with contraction patterns from cmap
    Input:
        sTxt: input string with contractions that need expansion
    Return:
        sTxt with expanded contractions    '''

    # Search string of contractions: "(ain't|aren't|can't|can't've|'cause|...)"
    reSearch = '(%s)' % '|'.join(ContractionsMap.keys())
    cre = re.compile(reSearch)  # compile regex search for speed

    def ReplaceMatches(match): 
    # retrieves matched expansion based on matched pattern group
        return cmap[match.group(0)]

    # substitute contraction matches with expansions:
    return cre.sub(ReplaceMatches, sTxt)


# Apply the specific set of rules to the `sTxt` to confirm that it expands the correct contraction.

# In[8]:


sTxt = "you're welcome in Ed's kitchen"
print(unContract_specific(sTxt))


# The [`contractions`](https://pypi.org/project/contractions/) package conveniently wraps these rules and provides some flexibility to add new ones. Caution:
# * a contraction can use [apostrophe](https://en.wikipedia.org/wiki/Apostrophe), [single quote](https://en.wikipedia.org/wiki/Quotation_mark#Summary_table), backquote, [grave mark](https://en.wikipedia.org/wiki/Grave_accent#Use_in_programming) and other similar-looking symbols. Some preprocessing may be needed to standardize all these marks.

# In[9]:


sTxt = "We're mining bitcoins on John's computer"
contractions.fix(sTxt)


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # Optional Practice
# 
# You will practice fixing the contractions in `sAlice` and the following string. 
# 

# In[ ]:


sTxt = "We're mining bitcoins on John's computer"


# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you've gotten the correct result. If you get stuck on a task, click the See **solution** drop-down menu to view the answer.

# ## Task 1
# 
# Fix contractions in `sTxt` by applying the `fix()` function from the contractions library.
# 
# <b>Hint:</b> This code is the same as the application of contractions package above.
# 

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details>
#         <summary><font color=#B31B1B>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# contractions.fix(sTxt)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# Use `nltk.word_tokenize()` to parse `sAlice` into a list of word tokens and save this as `LsAlice`. How many elements are in `sAlice`?
# 
# <b>Hint:</b> It's a simple application of <code>nltk.word_tokenize()</code>

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#B31B1B>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# LsAlice = nltk.word_tokenize(sAlice)
# len(LsAlice)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# Apply the `fix` method to `sAlice` to expand contractions, tokenize the string with `word_tokenize()`, and save the list of word tokens to `LsAliceCE`. How many elements are in this list? Since some words will be expanded into two or more words, you should be observing a larger count.
# 
# <b>Hint:</b> It's a simple application of <code>nltk.word_tokenize()</code>

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#B31B1B>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# LsAliceCE = nltk.word_tokenize(contractions.fix(sAlice))
# len(LsAliceCE)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 4
# 
# Apply `set()` to `LsAlice` list to remove duplicates and save this set of strings as `SsAlice`. What is the cardinality of this set (i.e., how many elements are in it)? 
# 
# <b>Hint:</b>  It's a simple application of <code>set()</code> to the list of strings.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details>
#         <summary><font color=#B31B1B>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# SsAlice = set(LsAlice)
# len(SsAlice)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 5
# 
# Similarly, apply `set()` to `LsAliceCE` to remove duplicates and save this set of strings to `SsAliceCE`. What is the cardinality of this set? 
# 
# <b>Hint:</b> It's a simple application of <code>set()</code> to the list of strings.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#B31B1B>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# SsAliceCE = set(LsAliceCE)
# len(SsAliceCE)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 6
# 
# Find all elements in `SsAliceCE` that are not in `SsAlice`. Your output should be word tokens that were not in the original text.
# 
# <b>Hint:</b> Use <code>.difference()</code> method of a set object.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#B31B1B>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# SsAliceCE.difference(SsAlice)
# SsAliceCE - SsAlice              # alternative notation for set difference
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 7
# 
# Find all elements in `SsAlice` that are not in `SsAliceCE` and save them to `LsOdd`. This odd output contains word tokens that are not in the preprocessed text because they were expanded. You should find 41 word tokens in the expanded text. 
# 
# <b>Hint:</b> Just like you did above, find the set difference in this task.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#B31B1B>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# LsOdd = SsAlice.difference(SsAliceCE)
# print(LsOdd)
# len(LsOdd)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 8
# 
# Take a closer look at `LsOdd`. Some of these elements are legitimate contractions that need expansion, but others are incomplete word parts. Why are these not appearing in `SsAliceCE` after expansion? This is an semi-open ended analytical question that requires you to investigate the text closer with the tools you have learned so far.
# 
# <b>Hint:</b> You can use <code>re.finditer</code> to find all match objects with starting and ending positions. Then you can offset these positions 10 characters wider and slice out a larger phrase containing the search pattern. Then tokenize this phrase with and without contraction expansion. You can place these operations in a loop for automation.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#B31B1B>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# sOddToken = "'sha"
# MO = re.finditer(sOddToken, sAlice) # match iterator
# sPhrases = [sAlice[max(0,mo.start()-10):min(len(sAlice), mo.end()+10)] for mo in MO]
# sPhrases
# print([nltk.word_tokenize(s) for s in sPhrases])
# print([nltk.word_tokenize(contractions.fix(s)) for s in sPhrases])
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 9
# 
# Find all unique tokens in `SsAliceCE` that still contain a single quote. How many are there?
# 
# <b>Hint:</b> Try a conditional set comprehension. In the condition check if <code>"'"</code> is in the token.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#B31B1B>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# SsContr = {s for s in SsAliceCE if "'" in s}
# len(SsContr)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 10
# 
# Find all tokens in `SsAliceCE` that still contain a single quote, but not as the first character of a token. Save this set of strings to `SsContrMid`. How many are there?
# 
# <b>Hint:</b> Same as above, but also check if the first character (at zero's position) is alpha or not. 

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#B31B1B>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# SsContrMid = {s for s in SsAliceCE if s[0].isalpha() and "'" in s}
# len(SsContrMid)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 11
# 
# Count number of instances of each element of `SsContrMid` in `sAlice` text. Think about ways you might want to improve your preprocessing pipeline to expand these contractions too.
# 
# <b>Hint:</b> Try <code>re.findall()</code> to find all instances of the search pattern.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#B31B1B>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# [(len(re.findall(s, sAlice)), s) for s in SsContrMid]
#             </pre>
#     </details> 
# </font>
# <hr>

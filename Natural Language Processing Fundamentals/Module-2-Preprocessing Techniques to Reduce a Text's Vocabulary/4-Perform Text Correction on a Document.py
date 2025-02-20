#!/usr/bin/env python
# coding: utf-8

# # Setup
# 
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import nltk, re


# To properly parse sentences with NLTK, you need to [download](https://www.nltk.org/data.html#installing-nltk-data) the punctuation corpus and [WordNet](https://en.wikipedia.org/wiki/WordNet) lexicon database.

# In[2]:


tmp = nltk.download(['punkt', 'wordnet', 'omw-1.4'], quiet=True) # download punctuations and WordNet database
from nltk.corpus import wordnet


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# ## Using Regex to Remove Duplicate Characters
# 
# The following `DedupTokens` function [recursively](https://en.wikipedia.org/wiki/Recursion_(computer_science)) removes duplicate characters (dups), until it reaches a word found in the WordNet database.

# In[3]:


def DedupTokens(LsTokens=['NNNo', 'Noooo', 'NoOoOoOo', 'Ann', 'Shall']):
    # pattern is precompiled for speed
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')  # find duplicated second group
    def replace(old_word):
        # recursively remove duplicated characters until the word is found in WordNet lexicon
        if wordnet.synsets(old_word): return old_word
        # return groups 1,2,3 only (without a duplicate of the group \2):
        new_word = repeat_pattern.sub(r'\1\2\3', old_word) 
        return replace(new_word) if new_word != old_word else new_word  # stopping criterion
    return [replace(word) for word in LsTokens]  # fix each word in the list

DedupTokens()


# Here's how the `DedupTokens` function works in more detail:
# 
# 1. The regex pattern `(\w*)(\w)\2(\w*)` looks for three capturing groups of word characters (`\w`:={letters, digits, _}) 
#     1. [`re.compile`](https://docs.python.org/3/library/re.html#re.compile) checks the regex argument and prepares it for reuse. This speeds up the multiple applications of the given regex.
# 1. The regex pattern `\2` matches a duplicated character of the second group, `(\w)`
# 1. `r'` indicates a raw string so that `\` characters are taken literally and not escaped by Python. These slashes are passed to regex, which uses them as escapes.
# 1. `replace()` is a helper function which recursively calls itself
# 1. `wordnet.synsets(old_word)` checks WordNet database for the existence of `old_word` in any letter casing
#     1. Thus, `wordnet.synsets('car')` and `wordnet.synsets('Car')` return the same result (which we will examine later)
# 1. The main function evaluates `replace()` on each token in `LsTokens`
#     1. `replace()` checks if the argument `old_word` is found in WordNet, which marks the end of deduping
#         1. Or, it removes the `(\w)` character from the word by leaving only the `\1\2\3` pattern in the new word
#     1. If no character was removed (i.e., no dup found), then we are done with deduping of the current word
#         1. Or, we call replace again to check on other dup characters
# 
# You can try [this regex in regex101](https://regex101.com/r/NwlUlO/1), an interactive online regex tool, which visually explains regex processing on a test string.
# 
# ## Limitations of the `DedupTokens()` Function
# 
# The `DedupTokens()` function is not ideal. In fact, rule-based algorithms will rarely  handle all scenarios perfectly (whatever "perfect" means here). Some limitations of this function are, for example: 
# 
# 1. It looks for a dup character in the exact same letter casing. We can improve the function by adding a regex parameter [`re.IGNORECASE`](https://docs.python.org/3/library/re.html#re.IGNORECASE) used to switch between case sensitive and case insensitive matches. 
# 1. It greatly depends on WordNet. New words, foreign words, identifying words (model numbers, phone numbers, etc.) may contain duplicate characters, but still be correct. For example, stock ticker `'AAA'`, `'BBB'`; better business bureau (`'BBB'`) organization, phone number `'800-555-7788'`, etc. Even the name `'Ann'` and the verb `'shall'` are not found in WordNet.
#     1. We can alleviate this issue by expanding the lexicon via addition of other relevant corpora.
# 
# To see an example of this, let's use `DedupTokens` on a few words. Here the correct word `'subbookkeeper'` is fixed incorrectly because it is not found in WordNet.

# In[ ]:


DedupTokens(['bittter', 'bassoonn', 'bookkeeper', 'subbookkeeper'])


# `DedupTokens()` works on most of the strings, but a correct word `'subbookkeeper'` is fixed incorrectly because it is not found in WordNet.
# 
# Next, parse a sentence into words and then apply `DedupTokens()` to the list of parsed tokens.

# In[ ]:


sPhrase = 'Learning at eCornell and Cornell is realllllyyy amaaazingggg'
sFixedPhrase = DedupTokens(nltk.word_tokenize(sPhrase))
' '.join(sFixedPhrase)


# In this instance, the incorrect words are fixed and the correct words remain correct. 

# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # Optional Practice
# 
# In these practice tasks, you will modify the `DedupTokens()` function. 
# 
# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you've gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.

# ## Task 1
# 
# Modify `DedupTokens()` so that it accepts a new boolean argument `IgnoreCase`. Use this argument to determine if you will need to use the [`re.IGNORECASE`](https://docs.python.org/3/library/re.html#re.IGNORECASE) flag in [`re.compile()`](https://docs.python.org/3/library/re.html#re.compile) to ignore letter casing in pattern matching. Name the new user defined function (UDF) as `DedupTokens2()` and test your function with the default parameter `LsTokens`.
# 
# <b>Hint:</b> Use the <code>flags</code> parameter of the <a href="https://docs.python.org/3/library/re.html#re.compile"><code>re.compile()</code></a>

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# def DedupTokens2(LsTokens=['NNNo', 'Noooo', 'NoOoOoOo', 'Ann', 'Shall'], IgnoreCase=False):
#   # pattern is precompiled for speed
#   flags = re.IGNORECASE if IgnoreCase else 0  # integer
#   repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)', flags=flags)  # find duplicated second group
#   def replace(old_word):
#     # recursively remove duplicated characters until the word is found in WordNet lexicon
#     if wordnet.synsets(old_word): return old_word
#     # return groups 1,2,3 only (without a duplicate of the group \2):
#     new_word = repeat_pattern.sub(r'\1\2\3', old_word) 
#     return replace(new_word) if new_word != old_word else new_word  # stopping criterion
#   return [replace(word) for word in LsTokens]  # fix each word in the list
# print(DedupTokens2(IgnoreCase=False))
# print(DedupTokens2(IgnoreCase=True))
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
#  
# <span style="color:black"> Use `nltk.download()` to load the `'gutenberg'` corpus and load the set of all unique words from `'carroll-alice.txt'` into the `SsAlice` variable. Apply `DedupTokens2()` to this set and wrap results as a set called  `SsAliceFixed`.
#     
# <b>Hint:</b> Use the <code>nltk.corpus.gutenberg.words()</code> method to load textbook from Gutenberg library. See examples <a href="https://www.nltk.org/book/ch02.html">here</a>.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# _ = nltk.download(['gutenberg'], quiet=True)
# LsAlice = nltk.corpus.gutenberg.words('carroll-alice.txt')
# SsAlice = set(LsAlice)
# SsAliceFixed = set(DedupTokens2(SsAlice))
# print(list(SsAliceFixed)[:20])
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# Print the words in `SsAlice` that were not fixed. 
# 
# <b>Hint:</b> You can do this with a <a href="https://docs.python.org/3/tutorial/datastructures.html#sets">set difference</a>.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# print(SsAlice - SsAliceFixed)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 4
#  
# You can improve this UDF check by using it on documents with additional lexicon. The Brown corpus uses a large vocabulary of English words.
#  
# 1. use `nltk.download()` to load the `'brown'` corpus. 
# 1. load a list of words from the corpus, then convert them to a set `SsBrO`
# 1. use set comprehension to lower-case these words and save to `SsBrLow` set of strings
# 1. union these two sets and save to `SsBrown` variable (as a set of strings)
#  
#  `SsBrown`$\leftarrow$ `SsBrO`$\cup$ lower-cased(`SsBrO`)
#  
# With this corpus, you now have a large vocabulary of lower- and original-cased words, which you can use to validate words in `replace()` helper function. Print the first twenty words from `SsBrown`.
# 
# <b>Hint:</b> You can convert all Brown words with <a href="https://docs.python.org/3/tutorial/datastructures.html#sets">set comprehension</a>.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# _ = nltk.download(['brown'], quiet=True)
# SsBrO = set(nltk.corpus.brown.words())  # original set of Brown words
# SsBrLow = {s.lower() for s in SsBrO}    # lower cased set of Brown words
# SsBrown = SsBrO.union(SsBrLow)    # both original and lower-cased
# print(list(SsBrown)[:20])
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 5
# 
# Validate that the original lexicon (SsBrO) contains 56,057 unique word tokens, the lower case lexicon (SsBrLow) contains 49,815 word tokens, and the final lexicon (SsBrown) contains 67,045 word tokens. Determine the number of words that are in mixed casing (and not in lower casing) in the Brown corpus.
# 
# <b>Hint:</b> Try a set difference.

# In[ ]:


# check solution here    


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# print('Original:', len(SsBrO), ', Lower:', len(SsBrLow), ', Final:', len(SsBrown), ', Added words:', len(SsBrO) - len(SsBrLow))
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 6
#  
# Find the first twenty (alphabetically sorted) words, which you created by lower-casing the original Brown words.
# 
# <b>Hint:</b> Try set difference and <a href="https://docs.python.org/3/howto/sorting.html#sorting-basics"><code>sorted</code></a> operation.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# print(sorted(SsBrO - SsBrLow)[:20])
#             </pre>
#     </details> 
# </font>
# <hr>

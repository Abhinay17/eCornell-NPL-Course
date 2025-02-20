#!/usr/bin/env python
# coding: utf-8

# # Part One of the Course Project
# 
# In this part of the course project, you will write code to complete function to split strings in a particular way, using the `split()` methods from both the standard Python library and the `re` library. As you determine how to best complete each function, carefully consider whether to use built-in string methods or regex methods. 
# 
# <hr style="border-top: 2px solid #606366; background: transparent;">
# 

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete this part of the course project. 

# In[1]:


get_ipython().run_line_magic('reset', '-f')
import nltk, re
from collections import Counter
import numpy.testing as npt, unittest
from typing import List
from colorunittest import run_unittest
eq, aeq = npt.assert_equal, npt.assert_almost_equal
tmp = nltk.download(['gutenberg'], quiet=True)        # See https://www.nltk.org/book/ch02.html
sRAW = nltk.corpus.gutenberg.raw('carroll-alice.txt') # string Raw Alice in Wonderland
sRAW[:200]
print(sRAW[:200])


# In[2]:


# Student: do not delete or modify this cell. It is used by autograder.


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Your Tasks**
# 
# There are 6 tasks for you to complete below.   
# 
# ## Checking Your Work
# 
# Test cases are provided below the code cell associated with each task. 
# 
# ## **Task 1**
# 
# Complete the `ParseOnWS()` function so that it:  
# - splits a string document by **whitespace** characters, i.e., ` `&nbsp;(space), `\t` (tab), `\n` (newline), `\r` (carriage return), into a list of string tokens. 
# - if specified, lowercases the string before tokenizing it so the returned list of string tokens is lowercased.
# 
# For reference, review these pages in the course:
# - Preprocess Substrings with Operations
# - Practice Preprocessing Substrings with Operations
# - String Manipulation Methods (downloadable Tool)

# In[3]:


# COMPLETE THIS CELL
def ParseOnWS(sDoc='Cats and dogs!', LCase=False)->List[str]:
    ''' Parse a string document on WHITESPACE using its split() method.
    sDoc (str): a document, which needs to be tokenized.
    LCase (bool): whether sDoc needs to be lower-cased before tokenization.
    Returns a list of string tokens from sDoc.
    Note: Newline characters (\n) should not generate list elements.'''
    # YOUR CODE HERE
    if LCase:
        sDoc = sDoc.lower()
    return sDoc.split()
    raise NotImplementedError()


# In[4]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestParseOnWS(unittest.TestCase):
    def test_00(self): eq(ParseOnWS(), ['Cats','and','dogs!'])
    def test_01(self): eq(ParseOnWS(LCase=True), ['cats','and','dogs!'])
    def test_02(self): eq(ParseOnWS('a\t \t\tb\nc\rd e !!'), ['a', 'b', 'c', 'd', 'e', '!!'])
    def test_03(self): eq(ParseOnWS(sRAW)[:5], ["[Alice's", 'Adventures', 'in', 'Wonderland', 'by'])
    def test_04(self): eq(ParseOnWS(sRAW, True)[:5], ["[alice's", 'adventures', 'in', 'wonderland', 'by'])
    def test_05(self): eq(len(ParseOnWS('My dog has fleas.\n')),4)


# ## **Task 2**
# 
# Complete the `ParseOnSP()` function so that it splits a `string` document into a list of string tokens on the **space** character ` `.
# 
# For reference, review these pages in the course:
# - Preprocess Substrings with Operations
# - Practice Preprocessing Substrings with Operations
# - String Manipulation Methods (downloadable Tool)

# In[5]:


# COMPLETE THIS CELL
def ParseOnSP(sDoc='Cats and dogs!', LCase=False)->List[str]:
    ''' Parse a string document on SPACE using its split() method.
    sDoc (str): a document, which needs to be tokenized.
    LCase (bool): whether sDoc needs to be lower-cased before tokenization.
    Returns a list of string tokens from sDoc.'''
    # YOUR CODE HERE
    if LCase:
        sDoc = sDoc.lower()
    return sDoc.split(' ')
    raise NotImplementedError()


# In[6]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestParseOnSP(unittest.TestCase):
    def test_00(self): eq(ParseOnSP(), ['Cats','and','dogs!'])
    def test_01(self): eq(ParseOnSP(LCase=True), ['cats','and','dogs!'])
    def test_02(self): eq(ParseOnSP('a\t \t\tb\nc\rd e  !!'), ['a\t', '\t\tb\nc\rd', 'e', '', '!!'])
    def test_03(self): eq(ParseOnSP(sRAW)[:5], ["[Alice's", 'Adventures', 'in', 'Wonderland', 'by'])
    def test_04(self): eq(ParseOnSP(sRAW, True)[:5], ["[alice's", 'adventures', 'in', 'wonderland', 'by'])


# ## **Task 3**
# 
# Complete the `ParseOnRE()` function so it splits a string document into a list of string tokens by the `[\s.!?:;"]` character class. Use the `re.split()` method and allow at least one repeat. 
# 
# For reference, review these pages in the course:
# - Parsing Strings with Regular Expressions
# - Practice Parsing Strings with Regular Expressions

# In[7]:


# COMPLETE THIS CELL
def ParseOnRE(sDoc='Cats and dogs!', LCase=False) -> List[str]:
    ''' Parse a string document on [\s.!?:;"] character class using re.split().
    sDoc (str):   a document, which needs to be tokenized.
    LCase (bool): whether sDoc needs to be lower-cased before tokenization.
    Returns a list of string tokens from sDoc'''
    # YOUR CODE HERE
    if LCase:
        sDoc = sDoc.lower()
    return re.split(r'[\s.!?:;"]+', sDoc)
    raise NotImplementedError()


# In[8]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestParseOnRE(unittest.TestCase):
    def test_00(self): eq(ParseOnRE(), ['Cats', 'and', 'dogs', ''])
    def test_01(self): eq(ParseOnRE(LCase=True), ['cats', 'and', 'dogs', ''])
    def test_02(self): eq(ParseOnRE('a\t \t\tb\nc\rd e  !!'), ['a', 'b', 'c', 'd', 'e', ''])
    def test_03(self): eq(ParseOnRE(sRAW)[:5], ["[Alice's", 'Adventures', 'in', 'Wonderland', 'by'])
    def test_04(self): eq(ParseOnRE(sRAW, True)[:5], ["[alice's", 'adventures', 'in', 'wonderland', 'by'])


# ## **Task 4**
# 
# Complete the `GetWords()` function so that it takes a string document and returns a list of string tokens extracted with a word character, `\w`. Use `re.findall()` function and allow at least one repeat. 
# 
# For reference, review these pages in the course:
# - Use Regular Expressions to Find Patterns
# - Practice Using Regular Expressions to Find Patterns
# - Parsing Strings with Regular Expressions
# - Practice Parsing Strings with Regular Expressions

# In[9]:


# COMPLETE THIS CELL
def GetWords(sDoc:str='Cats and dogs!', LCase=False)->List[str]:
    ''' Parse a string document to extract words using re.findall() and repeated \w pattern.
    sDoc (str): a document, which needs to be tokenized.
    LCase (bool): whether sDoc needs to be lower-cased before tokenization.
    Returns a list of tokens from sDoc'''
    # YOUR CODE HERE
    if LCase:
        sDoc = sDoc.lower()
    return re.findall(r'\w+', sDoc)
    raise NotImplementedError()


# In[10]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestGetWords(unittest.TestCase):
    def test_00(self): eq(GetWords(), ['Cats','and','dogs'])
    def test_01(self): eq(GetWords(LCase=True), ['cats','and','dogs'])
    def test_02(self): eq(GetWords('a\t \t\tb\nc\rd e  !!'), ['a', 'b', 'c', 'd', 'e'])
    def test_03(self): eq(GetWords(sRAW)[:5], ['Alice', 's', 'Adventures', 'in', 'Wonderland'])
    def test_04(self): eq(GetWords(sRAW, True)[:5], ['alice', 's', 'adventures', 'in', 'wonderland'])


# ## **Task 5**
# 
# Complete the `GetLex` function so that it retrieves an alphabetically sorted list of unique tokens from the a string document. Use `GetWords()` to tokenize the document.
# 
# For an example of sorting and an example of returning unique items, refer to the three practice tasks in the notebook on this page, found in Module 2:
# - Parsing a Document into Tokens

# In[11]:


# COMPLETE THIS CELL
def GetLex(sDoc:str='S t a t i s t i c s ', LCase=False)->int:
    ''' Parse a string document with GetWords(), remove duplicates and order alphabetically.
    sDoc (str): a document, which needs to be tokenized.
    LCase (bool): whether sDoc needs to be lower-cased before tokenization.
    Returns a list of ordered unique words from sDoc'''
    # YOUR CODE HERE
    words = GetWords(sDoc)
    if LCase:
        words = [word.lower() for word in words]
    unique_sorted_words = sorted(set(words))
    return unique_sorted_words
    raise NotImplementedError()


# In[12]:


# RUN CELL TO TEST YOUR CODE
sPoem = '"Rage, rage against the dying of the light." -- Dylan Thomas'
@run_unittest
class TestGetLex(unittest.TestCase):
    def test_00(self): eq(GetLex(), ['S', 'a', 'c', 'i', 's', 't'])   # unique characters
    def test_01(self): eq(GetLex(LCase=True), ['a', 'c', 'i', 's', 't'])
    def test_02(self): eq(GetLex(sPoem), ['Dylan', 'Rage', 'Thomas', 'against', 'dying', 'light', 'of', 'rage', 'the'])
    def test_03(self): eq(GetLex(sPoem, True), ['against', 'dying', 'dylan', 'light', 'of', 'rage', 'the', 'thomas'])
    def test_04(self): eq(GetLex(sRAW[:50]), ['Adventures', 'Alice', 'Carroll', 'Lewis', 'Wonderland', 'by', 'in', 's'])
    def test_05(self): eq(GetLex(sRAW[:50], True), ['adventures', 'alice', 'by', 'carroll', 'in', 'lewis', 's', 'wonderland'])


# ## **Task 6**
# 
# Complete the `GetTokFreq()` function so that it returns a list of counts in decreasing-order, from the most frequent words to the least frequent words in a string document. Use `GetWords()` to tokenize the document.
# 
# For reference, review these pages in the course:
# - Count Substrings
# - Practice Counting Substrings

# In[13]:


# COMPLETE THIS CELL
def GetTokFreq(sDoc:str='S t a t i s t i c s ', LCase=False, n=20)->List[int]:
    ''' Parse sDoc with GetWords() and compute word frequencies using collections.Counter().
    sDoc (str): a document, which needs to be tokenized.
    LCase (bool): whether sDoc needs to be lower-cased before tokenization.
    Returns a list of decreasing counts for top n most frequent words in sDoc'''
    # YOUR CODE HERE
    words = GetWords(sDoc)
    if LCase:
        words = [word.lower() for word in words]
    word_counts = Counter(words)
    most_common_counts = [count for word, count in word_counts.most_common(n)]
    return most_common_counts
    raise NotImplementedError()


# In[14]:


# RUN CELL TO TEST YOUR CODE
sPoem = '"Rage, rage against the dying of the light." -- Dylan Thomas'
@run_unittest
class TestGetTokFreq(unittest.TestCase):
    def test_00(self): eq(GetTokFreq(), [3, 2, 2, 1, 1, 1])   # most frequent characters
    def test_01(self): eq(GetTokFreq(LCase=True), [3, 3, 2, 1, 1])
    def test_02(self): eq(GetTokFreq(sPoem), [2, 1, 1, 1, 1, 1, 1, 1, 1])
    def test_03(self): eq(GetTokFreq(sPoem, True), [2, 2, 1, 1, 1, 1, 1, 1])
    def test_04(self): eq(GetTokFreq(sRAW[:50]), [1, 1, 1, 1, 1, 1, 1, 1])


# In[ ]:





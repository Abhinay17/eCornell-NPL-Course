#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# Review the code Professor Melnikov used to manipulate strings in the previous video. 
# 
# ## **String Basics**
# 
# Python [strings](https://docs.python.org/3/tutorial/introduction.html#strings) are defined by enclosing sequences of characters with a pair of single or double quotes. 

# In[2]:


sDoc1 = 'NLP is so fun.'    # we use single quotes to define a string of characters
sDoc1


# In[3]:


sDoc2 = "I like it a ton."  # we use double quotes to define a string of characters
print(sDoc2)


# You can format strings with invisible characters, which are special [escape characters](https://docs.python.org/3/reference/lexical_analysis.html#escape-sequences) that tell the Python interpreter to treat a character as special formating code, rather than as a string character. 
#     
# For example, you can define [multi-line strings](https://docs.python.org/3/tutorial/introduction.html#strings) by ending each new line with the invisible newline character, `\n`. The formatting of multi-line strings appears in the output when an object with multiple lines if printed with the `print()` function.

# In[4]:


sDoc3 = '''It is having
a good 
run.'''
print(sDoc3)  # formats the string with multiple lines, one per each \n character


# Some other invisible characters you can use to format strings include tab (`\t`), carriage return (`\r`) (typical to Microsoft Windows operating systems), form feed (`\f`), and vertical tab (`\v`). The distinction between invisible and regular characters is important when you work with regular expressions, as you will discover later in this module. 
#     
# When you don't use `print()` to display a string, invisible characters are displayed in the output cell as part of an unformatted string.

# In[5]:


sDoc3  # prints a string in a single line with \n characters


# The `str()` function converts nearly any Python object to a string, including numbers, lists, `None` values, function definitions, and other Python objects. 

# In[6]:


str('a'), str(1), str([3,4]), str(print)


# The `str` function can also accept string inputs, with the following functionality:  
# * `str(str)` converts `str` (a class definition in Python) to the string `<class 'str'>`. 
# * `str(str(1))` results in the same output as `str(1)`.

# In[7]:


str(None), str(str), str(str(1))


# ## Splitting and Slicing Strings
# 
# A simple way to split a string into characters is to apply the `list()` function to the string.

# In[8]:


print(list(sDoc1))


# Notice that every character (even a space) of the original string, `sDoc1`, becomes an element of a list.
# 
# **Slicing** or **subsetting** allows you to extract substrings using their *integer-indexed positions*. Indexing in Python starts from 0 rather than 1, so the slice `[0:3]` references the indices 0, 1, and 2 in the given list (or tuple or array), but ignores index 3. This is useful because it means that to partition the string into two non-overlapping substrings, you can indicate slices `[0:n]` and `[n:]`, where the character at the nth position falls into the second substring.
# 
# A negative index counts characters from the end of a string. Examine some different slices of the `sDoc1` string, 'NLP is so fun'. 
# 

# In[9]:


sDoc1, sDoc1[0:3], sDoc1[:3], sDoc1[4:6], sDoc1[-4:]


# ## Escaping and printing literal characters

# In Python, the backslash (`\`) character is used to assign a special meaning to some characters. For example, `\n` is interpreted as a newline character, which forces the string to break to a new line. To force the original (literal) meaning to `\n`, i.e., the backslash and the character `n`, you should escape the escape as `'\\n'` or, alternatively, use the **raw string** by prepending the letter `r` before the string, i.e., `r'\n'`. The latter is equivalent to double-backslashing and is a convenient shortcut when many special characters need to be reverted to their original literal meaning.
# 
# The examples below demonstrate the differences in outputs of escaped and non-escaped characters.

# In[10]:


'0\n1' # string is displayed literally (without print())


# In[11]:


print('0\n1') # newline is printed as an invisible break between lines


# In[12]:


'0\\n1' # string is displayed literally (without print())


# In[13]:


# characters '\' and 'n' are literaly printed, not newline character
print('0\\n1') 
print(r'0\n1')


# The following example checks which statements contain a backslash character.

# In[14]:


print('\\')  # confirm that this is a literal backslash character
print('\\' in '\n') 
print('\\' in '\\n')
print('\\' in r'\n')


# ## Other string prefixes

# Here is a more complete list of string prefixes
# 
# 1. `r'...'` defines a raw string to keep escape characters as literals.
# 1. `f'...'` defines a [formatted string](https://docs.python.org/3/reference/lexical_analysis.html#f-strings) (or f-string), which allows for a quick formatting of numbers, dates, and other data types using `'{}'` notation.
# 1. `u'...'` defines a [unicode string](https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals), which is just an [ordinary string](https://peps.python.org/pep-0414/#proposal), contrary to ASCII string, which disallows unicode characters (such as non-Latin alphabet characters). In other words, all strings in Python are unicode strings by default, so the use of `u` prefix is redundant.
# 1. `b'...'` defines a [byte string](https://docs.python.org/3/library/stdtypes.html#bytes), which can be stored on disk without an additional encoding. Since a computer stores and operates on bytes (consisting of 8 bits), i.e. numbers 0-255, most data structures require encoding to convert them to bytes and decoding to convert them to "human-readable" format.
# 
# A few examples below demonstrate these special strings.

# In[15]:


print(f'1/7 ≈ {1/7:.4f}')      # rounds 0.14285714285714285... to 4 decimals
print(u'abc'=='abc')           # shows that strings are unicode by default in Python 3.x
print(b'this is byte string')  # only ASCII characters are allowed and assumed in bytestrings


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Optional Practice**
# 
# Now you will practice some of these basic string manipulation techniques by working with the name of the bacteria *Myxococcus llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogochensis*.

# In[3]:


sTxt = 'The quick brown fox jumped Over the Big Dog And Then Jumped Again Over The Lazy Myxococcus llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogochensis'
sTxt


# This long, rather silly string contains a nice mix of features for you to work with: capital and lowercase letters, words of different lengths, and spaces between words.
#     
# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer. </span>

# ## Task 1
# 
# Use the `list()` function to split `sTxt` into characters and print out the list contents. 
# 
#   **Hint:** Call the `list()` function with `sTxt` as its argument.

# In[18]:


print(list(sTxt))# check solution here


# <font color=606366>
#     <details><summary><font color=B31B1B>▶ </font>See <b>solution</b>.</summary>
#             <pre class="ec">
# print(list(sTxt))
#             </pre>
#         </details>
#     <hr>

# ## Task 2
# 
# Slice `sTxt` so that you return its first three characters as a string object. Save the result as a new variable, `sThe`.
# 
# **Hint:** Use slice [:3] or [0:3] on the string `sTxt`.

# In[ ]:


sTxt[:3]# check solution here


# <font color=606366>
#     <details><summary><font color=B31B1B>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# sThe = sTxt[:3]
# sThe = sTxt[0:3]
#             </pre>
#         </details>
# <hr>

# ## Task 3
# 
# Index or slice the string so that you return only the letter `'e'` from the string `sThe`.
# 
# **Hint:** You can use -1 index to slice from the right. For example, [-1:] returns the rightmost character of the string to which this slice is applied.

# In[ ]:


# check solution here


# <font color=606366>
#     <details><summary><font color=B31B1B>▶ </font>See <b>solution</b>.</summary>
#             <pre>
#   sThe[-1]
#   sThe[-1:]
#   sThe[2:3]
#   sThe[2]
#             </pre>
#         </details>
# </font>
# 
# <hr>

# ## Task 4
# 
# Slice `sTxt` so that you return the seventh character from the right. It should be the letter `'c'`.
# 
# **Hint:** You can slice a string with right indexing using a minus sign, or with left indexing using the length of the string.

# In[ ]:


# check solution here


# <font color=606366>
#     <details><summary><font color=B31B1B>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# sTxt[-7:-6]   # An easy approach is to use right-indexing
# # Another approach is to left-indexing, but you need to know full length of the string. 
# n = len(sTxt) # Calculate the length, n, with len()
# sTxt[n-7:n-6]
# # An even simpler solution is:
# sTxt[-7]
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 5
# 
# Slice `sTxt` to retrieve the full bacteria name, and save it as a new variable, `sBacteria`.
# 
# **Hint:** Try several left indices until you find one that retrives the bacteria name.

# In[4]:


sBacteria = sTxt[-74:]
sBacteria


# <font color=606366>
#     <details><summary><font color=B31B1B>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# # Solution 1: try several indices to get to the correct one
# sBacteria = sTxt[-74:]
# sBacteria
# # Solution 2: split sentence into words and join the last two words with a space
# sBacteria = " ".join(sTxt.split(' ')[-2:])
# sBacteria
# # Solution 3: find index of 'M' to perform the slice
# pos = sTxt.index('M')
# sBacteria = sTxt[pos:]
# sBacteria
#             </pre>
#         </details>
# </font>
# 
# <hr>

# ## Task 6
# 
# Print a list of the index positions of all occurrences of the letter `'l'` in `sBacteria`.
# 
# **Hint:** You can iterate over characters while incrementing a counter `'l'`. The `'l'` is added to the list whenever `'l'` is encountered. A different approach is to iterate over characters via list comprehension. The iterator here can be created with the <a href="https://docs.python.org/3/library/functions.html#enumerate"><code>enumerate()</code></a> function.
# 

# In[5]:


i = 0 # incrementing variable
LnPos = []   # list of positions
for c in sBacteria:
    if c == 'l':
        LnPos += [i]
    i += 1
LnPos# check solution here


# <font color=606366>
#     <details><summary><font color=B31B1B>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# # Solution 1: 
# i = 0 # incrementing variable
# LnPos = []   # list of positions
# for c in sBacteria:
#     if c == 'l':
#         LnPos += [i]
#     i += 1
# LnPos
# 
# \# Solution 2: 
# [i for i,c in enumerate(sBacteria) if c=='l'] # a more compact solution using list comprehension
# 
# \# Solution 3:
# import numpy as np    # an even faster solution requires you to import NumPy
# sBacArray = np.array(list(sBacteria))
# np.where(sBacArray == 'l')
#            </pre>
#         </details>
# </font>
# 
# 

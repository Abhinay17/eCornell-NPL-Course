#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# 
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import re, pandas as pd, unicodedata


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# Python supports different types of strings, including raw, f-string, binary, and unicode strings. In particular, a [unicode string](https://www.python.org/dev/peps/pep-0414/#proposal) is defined by a unicode (`u`) character that precedes a string, e.g., `u'NLP'`. In Python 3.x, all strings are unicode by default, so the `u` decorator is not necessary. Unicode strings can store non-[ASCII](https://en.wikipedia.org/wiki/ASCII) characters, such as accented letters, symbols, and even emojis.
# 
# ## Reducing Text Vocabulary by Normalizing Strings
# 
# Stripping accent marks from characters is an important preprocessing technique because it allows you to reduce the vocabulary of your document. Unfortunately, simply converting your text from unicode to ASCII with the `str.encode()` method does not de-accent letters, and instead removes the accented letters.

# In[2]:


s = u'A sugar-free crème brûlée is '  'still a creme brulee and costs $1,234.777!!! 🆆🅾🆆 😋🍮'
s.encode('ascii', 'ignore')


# A safer approach is to first normalize the string with the [`unicodedata.normalize()`](https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize) method and specifying the desired normalization form. Below, we use the `'NFKD'` form of normalization. 
# 

# In[3]:


normalized = unicodedata.normalize('NFKD', s)
normalized


# Once normalized, the unicode string can be converted to ASCII, and the accented characters are replaced with their closest ASCII equivalent. Thus, `'û'` is replaced with `'u'` and so on. Characters without an equivalent are deleted from the output string. 

# In[4]:


encoded = normalized.encode('ascii', 'ignore')
encoded


# Finally, decode the encoded string.

# In[5]:


encoded.decode('utf-8', 'ignore')


# ## Removing All Special Characters 
# 
# Sometimes you will need to remove all special characters. You should be careful to avoid corrupting phrases and structured expressions. Accommodating all edge cases can require a complex regex expression. Note that the `'\w'` word character treats unicode letters like any other letter, but `'A-Z'` are strictly ASCII letters.

# In[6]:


print(re.sub(pattern='[^A-Za-z_ ]+', repl='', string=s))
print(re.sub(pattern='[^\w ]+', repl='', string=s))


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # Optional Practice
# 
# You will standardize some sentences using the regex and normalization techniques discussed above. Run the following code to load and print the strings you will work with. 

# In[7]:


sSAT = 'Sómě Áccěntěd těxt'
sStd = 'élève, Elève, élEvé, élévé, éleve, eLevé, eléVe'  # "Student" in French
sPhone = "1 (123) 345-6789 Jack"  # Goal: "1 (123) 345-6789 Jack" ➞"11233456789"
sPost = "In 2018, I made my first crème brûlée! Did you like it? :), ;-), *&@)#@!" 


# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you've gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.

# ## Task 1
# 
# Complete the `NumbersOnly()` function so that it removes all non-digit characters from a string. Then, call this function with `sPhone` and `sPost` variables.
# 
# <b>Hint:</b> Try <code>re.sub()</code> with character class <code>[^0-9]</code>

# In[ ]:


def NumbersOnly(sTxt='') -> str:
    ''' Removes all non-digit characters from a string
        Returns: the digit characters from sTxt'''
    # check solution here

    return sTxtNumOnly


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# # Solution 1:
# def NumbersOnly(sTxt='') -> str:
#     sTxtNumOnly = re.sub(pattern = r'[^0-9]', repl = '', string = sTxt)
#     return sTxtNumOnly
# 
# NumbersOnly(sPhone)
# NumbersOnly(sPost)
# 
# <span># Solution 2:</span>
# def NumbersOnly(sTxt='') -> str:
#     sTxtNumOnly = re.sub(pattern = r'[^\d]', repl = '', string = sTxt)
#     return sTxtNumOnly
#     
# NumbersOnly(sPhone)
# NumbersOnly(sPost)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# Now, complete the `AlphaNumOnly()` function, which takes a string and drops all non-alphanumeric ASCII characters (i.e., upper and lower a-z and digits 0-9). Then use it to clean the variables `sPhone` and `sPost`.
# 
# <b>Hint:</b> Try <code>re.sub()</code> with character class <code>r'[^a-zA-Z0-9]'</code>

# In[ ]:


def AlphaNumOnly(sTxt='') -> str:
    ''' Removes all non-alphanumeric characters from a string
        Returns: the alphanumeric characters from sTxt'''
    # check solution here

    return sTxtAlphaNum


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>
#             See <b>solution</b> 
#         </summary>
#             <pre>
#             
# def AlphaNumOnly(sTxt='') -> str:
#     sTxtAlphaNum = re.sub(pattern = r'[^a-zA-Z0-9]', repl = '', string = sTxt)
#     return sTxtAlphaNum
# 
# AlphaNumOnly(sPhone)
# AlphaNumOnly(sPost)
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# Create a function called `NoAccents()` that takes a string and both replaces accented characters with their ASCII equivalents **and** lowercases all letters. Apply this function to `sSAT` and `sStd` variables.
# 
# <b>Hint:</b> Try the same normalization as above, but also call <code>.lower()</code> method on the resulting output.

# In[ ]:


def NoAccents(sTxt='') -> str:
    ''' Replaces accented characters in sTxt with their ASCII equivalents
        Lowercases all letters
        Returns: a lowercased sTxt with no accents'''
    # check solution here

    return sTxtNoAccents            


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>
#             See <b>solution</b> 
#         </summary>
#             <pre>
# def NoAccents(sTxt='') -> str:
#     sTxtNoAccents = unicodedata.normalize('NFKD', sTxt).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower()
#     return sTxtNoAccents
#     
# NoAccents(sSAT)
# NoAccents(sStd)
#             </pre>
#     </details> 
# </font>

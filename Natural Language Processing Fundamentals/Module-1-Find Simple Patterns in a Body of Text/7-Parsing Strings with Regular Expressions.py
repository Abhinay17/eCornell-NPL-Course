#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.  

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import re, pandas as pd


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # Review
# 
# Review the code Professor Melnikov used to parse strings with regexes in the previous video
# 
# ## Compare `re.findall()` and `re.split()`
# 
# Previously we used [`re.findall()`](https://docs.python.org/3/library/re.html#re.findall) to find pattern matching words. This can be generalized to find all words in a sentence. Essentially, this is parsing a sentence into word tokens. Alternatively, we can use [`re.split()`](https://docs.python.org/3/library/re.html#re.split) to split string on the `\W+` regex pattern to find all word tokens. It splits the string on any non-word characters, meaning anything that isn't a letter, digit, or underscore.

# In[2]:



sFox = 'The quick-brown fox jumps over the lazy_dog...'
print(re.findall('\w+', sFox))     # split on at least one contiguous word character
print(re.split('\W+', sFox))       # split on at least one contiguous non-word character


# Notice the extra empty string found by `re.split()`. Non-word characters such as `'...'` at the end of the string `s` separate `'lazy_dog'` from the empty string `''`. There are many ways to fix this, but the key takeaway is to always evaluate your results after applying any string processing method and note any unusual behavior.
# 
# You could also tokenize a sentence into words with a [character class](https://www.regular-expressions.info/charclass.html), defined by square brackets `[]`, to find words between spaces and punctuation. For this method to be successful, the character class needs to list all word-separating characters in the text.

# In[3]:


s = 'NLP is gr8! Python-3 is A1.'
print(re.findall('[^ !.]+', s))  # split on at least one contiguous character other than space, ! or a period
print(re.split('[ !.]+', s))     # split on at last one contiguous space or ! or a period


# In the example above, just space (` `), `!`, and `.` are sufficient to identify words. 
# 
# ## Parsing Words that Contain Periods
# 
# If some of the words in your document include periods, forcing a period+space combination may be necessary so words that contain periods aren't split. The word pattern `'. '` is often used to separate sentences. 

# In[4]:


s = 'NLP is gr8! Python-3.x is A1.'
print(re.findall('[\w -]+', s))   # split on at least one contiguous word character r space or -
print(re.split('[.!?]+', s))      # split on at least one period or ! or ?
print(re.split('[.!?] +', s))     # split on a single character in character class followed by a space


# However, be cautious because some sentences may lack a period or a space. For example, menu items in Wikipedia article are period-less, yet still can be considered as sentences or independent phrases, so they would need individual processing.

# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # Optional Practice
# 
# Let's continue with a bit of practice of our own. 
# 
# Start with the [presidential oath](https://constitution.congress.gov/browse/essay/artII-S1-C8-1/ALDE_00001126/), which we will parse into individual words and sentences.
# 
# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.

# In[5]:


sOath = 'Article II, Section 1, Clause 8:\n\nBefore he enter on the Execution of his Office, he shall take the following Oath or Affirmation:\n–I do solemnly swear (or affirm) that I will faithfully execute the Office of President of the United States, and will to the best of my Ability, preserve, protect and defend the Constitution of the United States.'
print(sOath)
sOath


# ## Task 1
# 
# Split oath at the colon character and return a list of 3 individual sentences.
# 
#  <b>Hint:</b> Try splitting on <code>':'</code>

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1>▶</font>
#             See <b>solution</b>.</summary>
#             <pre>
# re.split(':', sOath)     # solution 1: split oath at the colon character
# re.split(r':+', sOath)   # solution 2: split oath at the colon character
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 2 
# 
# Parse oath into word tokens (which consist of any number of word characters) and return their count. 
# 
# <b>Hint:</b> Try finding all contiguous <code>'\w'</code> characters, which will essentially split on non-word characters.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# len(re.findall('\w+', sOath)) # find number of words in sOath
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 3
# 
# Split oath into individual words again, but this time identify the longest word and its count. 
# 
# <b>Hint:</b> Parse as you did above. Then compute lengths and use <code>sorted()</code> or <code>max()</code> methods to find the longest word. You can also use a Pandas DataFrame to accomplish these tasks.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# # find longest word and its length (in characters)
# LsWords = re.findall('\w+', sOath)
# 
# sorted([(w, len(w)) for w in LsWords], key=lambda w_len: w_len[1], reverse=True)  # solution 1
# 
# max([(w, len(w)) for w in LsWords], key=lambda w_len: w_len[1])  # solution 2; ?max to view help
# 
# df = pd.DataFrame(LsWords, columns=['word'])
# df['Len'] = df.word.str.len()
# df.sort_values('Len').tail(1).values.ravel().tolist()      # solution 3
# 
# df.iloc[df.Len.idxmax(1)].values.tolist()                  # solution 4
# 
# ls = re.findall('\w+', sOath)                              # solution 5
# res = [len(i) for i in ls]
# ls[res.index(max(res))], max(res)
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 4
# 
# Find all non-overlapping neighboring word pairs (i.e., 2-grams) separated by a space, where a word is a sequence of word characters, `\w`, between non-word characters, `\W`, or string start/end.
# 
# So, `'Article II'` qualifies because `'Article'` and `'II'` are space-separated words, but the 2-gram `'II, Section'` does not because these words are separated by `', '` and not by a space. 
# 
# <b>Hint:</b> Try using a regex string with two <code>'\w+'</code> search patterns with a space in between

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# # find word pairs separated by a space
# re.findall('\w+ \w+', sOath)
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 5
# 
# 
# Find all space-separated word pairs again (as in Task 4) with the first word starting with the letter `'o'` and the second word starting with the letter `'t'` or `'o'`. Ignore letter casing.
# 
# <b>Hint:</b> Same as above, but you need to add starting letters to the pattern and word boundaries to ensure only one starting letter is considered. 

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# # find all case-ignorant space-separated word pairs, 
# # where first word starts with 'o' and second word starts with 't'
# # Solution 1 with double escape characters (without raw string)
# re.findall('\\bo\\w+ \\b[ot]\\w+', sOath, flags=re.IGNORECASE)
# #
# # Solution 2 with raw string. Recall (from the *Practice Parsing Strings with Regular Expressions* in the first Jupyter Notebook (JN)  that `r'...'` is a raw string which cancels the effect of the escape character `\`.
# re.findall(r'\bo\w+ \b[ot]\w+', sOath, flags=re.IGNORECASE)
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 6
# 
# 
# Find all space-separated word pairs again, this time with the first word being one character long. Ignore letter casing.
# 
# <b>Hint:</b> You can use a single word character search to find one-letter words.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# # find all case-ignorant space-separated word pairs, where first word has a single letter
# re.findall(r'\b\w \b\w+', sOath, flags=re.IGNORECASE)
#             </pre>
#         </details>
# </font>
# <hr>

# ## Parsing Lists
# 
# 
# In this example we will apply a parsing method to a list of email addresses. Parsing lists containing semi-structured elements is a common task. Such lists can contain addresses (email, home, IP), numeric identifiers (phone numbers, student IDs, social security numbers), login names (SkypeID, Facebook ID), or even short messages (tweets, reviews).

# In[ ]:


sClass = '<Alex> KeepOnLearning@eCornell.com; <Anna> LifeLongLearner@outlook.com; <Atya> Student777@gmail.com; <Alice> ScienceGr8@Cornell.edu; '


# ## Task 7
# 
# Return a list of 4 students (names and emails), i.e., parse the list at characters separating the students' information.
# 
# <b>Hint:</b> Try splitting on <code>'; '</code>. You might need to remove one empty string element from the resulting list to return student names/emails only. In a bit more advanced solution you can remove the angle brackets also.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details><summary><font color=#b31b1>▶</font>See <b>solution</b>.</summary>
#             <pre>
# # your solution goes here
# LsStud = re.split('; ', sClass)[:-1]  # return a list of 4 students
# #--- Drop brackets in student names
# [s.replace('<','').replace('>', '') for s in LsStud] # solution 1
# [re.sub('[<>]','', s) for s in LsStud] # solution 2 via ReGex's sub
#             </pre>
#         </details>
# </font>
#         
# <hr>

# ## Task 8
# 
# Return a list of valid email addresses only.
# 
# <b>Hint:</b> Use a cominbation of word boundary, word characters, <code>'@'</code> symbol, and an escaped period character to construct the search pattern.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# re.findall(r'\b\w+@\w+\.\w+', sClass)  # return list of email addresses
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 9
# 
# Return a list of valid lower-cased email domains, such as `'Cornell.edu'`.
# 
# <b>Hint:</b> Similar to above, but your search pattern starts with characters following the @ symbol.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# re.findall(r'\b\w+\.\w+', sClass.lower())  # return list of email domain names. E.g. "gmail.com"
#             </pre>
#         </details>
# </font>
# <hr>

#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell


# <span style="color:black">You will be using [TextBlob](https://textblob.readthedocs.io/en/dev/), a popular NLP library, to correct misspellings. Many of its functions overlap with `nltk`, `Spacy`, `Gensim`, and other NLP libraries. For better integration, you will want to do as much as possible with the tools from the same library.

# In[2]:


from textblob import Word


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # Review
# 
# TextBlob's `Word` object behaves very much like a string when printed, concatenated with other strings, sliced, etc. However, it has additional complex methods that Python strings do not have. One method corrects misspellings using a popular [algorithm](https://norvig.com/spell-correct.html) created by [Peter Norvig](https://en.wikipedia.org/wiki/Peter_Norvig). 
# 
# Explore these functionalities by wrapping the misspelled word `'fianlly'` into a `Word` object.

# In[3]:


print(Word('fianlly'))
print(Word('fianlly')+'!')
print(Word('fianlly'[:3]))
print(Word('fianlly').correct())   # Peter Norvig's algorithm


# <span style="color:black">Peter Norvig's algorithm also calculates a standardized score, between 0 and 1, for the identified candidate(s). A higher score indicates a more likely candidate for the misspelled word. You can get these scores using the `spellcheck()` method.

# In[4]:


print(Word('fianlly').spellcheck())  # candidate & confidence score


# <span style="color:black"> Shorter words tend to have more candidates. Here is an example with multiple candidates.

# In[5]:


print(Word('teh').spellcheck())


# <span style="color:black"> To correct a sentence, you can tokenize the sentence, loop through the word tokens to correct any misspellings, then join the corrected words back together into a single string.

# In[6]:


sScrambled = '''Thea ordirng oof leeetters in a wrod ies noot imporant.'''
LsCorrected = [Word(s).correct() for s in sScrambled.split()]
print(' '.join(LsCorrected))


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Optional Practice**
# 
# You will now practice using `Word` objects.
#     
# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you've gotten the correct result. If you get stuck on a task, click the See **solution** drop-down menu to view the answer. You will need the following small helper function to find candidates or bets for the misspelled word. 

# In[7]:


def SpellBets(sScrambled='het'):
  'Prints a count and a list of candidates'
  LsCandidates = Word(sScrambled).spellcheck()  # find bets
  print(f'{len(LsCandidates)},', [w+f',{n:.3f}' for w,n in LsCandidates])
SpellBets('the')  # returns 1 candidate
SpellBets('eth')  # returns 6 candidates
SpellBets('het')  # returns 25 candidates


# ## Task 1
# 
# By exploring different permutations of letter positions only, scramble the spelling of `'junk'` (without dropping or introducing letters) to have more than twenty candidates. You can use the `SpellBets()` method for convenience.
# 
# <b>Hint:</b> Try permuting letters in some consistent manner to avoid confusion.

# In[8]:


SpellBets('nujk')


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# SpellBets('nujk')
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# By mixing the letter positions (except letter `'t'`), scramble the spelling of `'trash'` to have more than thirty candidates. You can use the `SpellBets()` for convenience.
# 
# <b>Hint:</b> To avoid confusion, try writing down your scrambled words and generate them by permuting letters in some consistent manner.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# SpellBets('thasr')
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# By mixing the letter positions (except first and last letters), scramble the spelling of `'garbage'` to have more than two candidates. You can use the `SpellBets()` for convenience. 
# 
# <b>Hint:</b> You might even find a scrambled version of the word with three candidates of corrected words: <code>['gargle,0.333', 'garage,0.333', 'barge,0.333']</code>. Now, can you guess the scrambled word?

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# SpellBets('gbargae')
#             </pre>
#     </details> 
# </font>
# <hr>

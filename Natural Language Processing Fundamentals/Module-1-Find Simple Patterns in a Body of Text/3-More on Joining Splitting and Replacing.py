#!/usr/bin/env python
# coding: utf-8

# # **Setup**
# 
# Work through the following examples to discover more ways you can use the `join()`, `replace()`, and `split()` methods to preprocess strings. 

# In[3]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell

# Create variables you'll use in this coding activity:
sDoc1 = 'NLP is so fun.'
sDoc2 = 'I like it a ton.'


# <h2>More methods for joining, replacing and splitting strings</h2>
# 
# The following cells contain more techniques you can perform with the [`join()`](https://docs.python.org/3/library/stdtypes.html#str.join), [`replace()`](https://docs.python.org/3/library/stdtypes.html#str.replace), and [`split()`](https://docs.python.org/3/library/stdtypes.html#str.split) methods. These techniques can be important when you preprocess strings, and you will use them as you work through the rest of this course. 

# ## `join()` method 
# 
# The `join()` method takes a list (or any iterable) of strings and concatenates (or "glues") them in the same order of a single larger string. If you attempt to pass a list of numbers or some other elements (which is a common mistake), an interpreter displays an error message. 

# In[4]:


sDoc3 = sDoc1 + ' ' + sDoc2      # concatenate strings
print(sDoc3)
print(' '.join([sDoc1, sDoc2]))  # the same output, but concatenation is done with join()


# The example below creates a list of odd integers `LnOdd`. If passed to `join()` directly, a `TypeError` is thrown. Instead, each element needs to be cast to a string as is shown below.

# In[5]:


LnOdd = [i for i in range(20) if i%2]   # conditional list comprehension resulting in a list of integers
LsOdd = [str(i) for i in LnOdd]  # converts each integer to a string
LnOdd, LsOdd


# You can catch a runtime error with [`try` clause](https://docs.python.org/3/tutorial/errors.html#handling-exceptions) to avoid interruption of interpreter's execution. In the cell below, two `join()` methods are called with different arguments. The first one throws an `TypeError`, while the second successfully executes joining all string characters into a single summation expression (still stored as a string).

# In[6]:


try:
    ' + '.join(LnOdd)
except BaseException as err:
    print(err, type(err))

' + '.join(LsOdd)


# Finally, the following statement displays the individual summands on the left of the equality sign and the corresponding total sum on the right hand side.

# In[7]:


' + '.join(LsOdd) + ' = ' + str(sum(LnOdd))


# ## `replace()` method
# 
# The `replace()` method allows you to search and replace an old string pattern with a new string pattern. 
# 
# The example below defines a string of nucleotides (characters A, C, T, G), `sNucleotides`, which we often call a DNA sequence. You can check if a specific subsequence is in the `sNucleotides` with an `in` operation. 

# In[8]:


sNucleotides = 'CTGAACTGAGACTTGGACTGAACTGACTGACTGACTGACTGACTGACTGACTGACTGACT'
'GACTTG1' in sNucleotides  # check whether 'GACTTG1' is in sNucleotides


# You can also search for a pattern `'. '` and replace it with an exclamation sign.

# In[9]:


sDoc4 = sDoc3.replace('. ', '!🎉 ')   # replace a period+space with new pattern
sDoc4


# In[10]:


sDoc1, sDoc1.replace(' ', '_')     # replace spaces with underscores


# ## `split()` method
# 
# The `split()` method divides the string into a list of substrings, which were originally separated by some pattern. In the example below, the string is split by spaces.

# In[ ]:


sDoc5 = sDoc1.split(' ')
sDoc5


# The new list of strings can now be joined with a new character, which produces a result similar to the one derived from the `replace()` function.

# In[ ]:


sDoc1, '_'.join(sDoc5)  # same result as above, sDoc1.replace(' ', '_')


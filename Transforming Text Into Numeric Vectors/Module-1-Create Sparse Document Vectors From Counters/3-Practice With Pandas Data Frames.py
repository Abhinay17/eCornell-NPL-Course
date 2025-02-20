#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video. 

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import pandas as pd, numpy as np


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# In this notebook, you'll practice creating and manipulating Pandas DataFrame objects and examine how they are similar to and differ from NumPy arrays. 
# 
# <h3 style="color:black"> Pandas DataFrames</h3>
# 
# In the next cell, you will use the [`.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) method to create a 5 $\times$ 10 list of 50 numbers ranging from 0 to 49. Then, you'll turn this list into a Pandas DataFrame object with the [`.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) method. The row indices are labeled with the `index` argument and column indices are labeled with the `columns` argument.

# In[2]:


Arr = np.reshape(list(range(50)),(5, 10))    # reshapes a 1D array of 50 values into 5x10 2D array
Arr
df = pd.DataFrame(Arr, index=list('VWXYZ'), columns=list('ABCDEFGHIK'))
df


# Just like with NumPy arrays, DataFrames have a [`shape`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html) method, which returns a tuple with row and column counts.

# In[3]:


df.shape  # returns # of rows and columns


# You can slice (or subset) columns and rows from a DataFrame object using their names in brackets. Notice that double-bracketed names return a Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) object, while a single-bracketed (single) name returns a Pandas [`Series`](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) object, which is similar but has fewer attributes and is meant for a single sequence of values, not an object in a tabular format.

# In[4]:


df['A']  # Returns a Series object with 1 column. Same as df.A


# In[5]:


df[['A']]  # Returns a DataFrame object with 1 column. 


# You can also use integer (zero-based) indices to slice columns and rows.

# In[6]:


df[0:2]  # returns 2 rows with indices 0 and 1, but index 2 is excluded


# A DataFrame's elements can store non-numeric values, such as strings, lists, and more complex objects.

# In[7]:


LsInt = dir(0)  # list of strings, names of attributes of an integer object
dfInt = pd.DataFrame(LsInt, columns=['Methods']) # dataframe with attributes names for an integer object
dfInt.T   # show a transposed (on its side) dataframe


# To compute the length of any string value in the [Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) object, you can call the [`str`](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.html) attribute to gain access to stored string values and then use the string's [`len()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.len.html) method. Store these results in a new column in the `dfInt` DataFrame.
# 

# In[8]:


dfInt['Length'] = dfInt['Methods'].str.len()   # retrieve length of each string in a dataframe and store in Legnth column
dfInt.T    # show dataframe transposed, so that it occupies less vertical space on screen


# You can compute basic statistics on any numeric columns using [`mean`](https://pandas.pydata.org/docs/reference/api/pandas.Series.mean.html) method and other statistical functions applied to the retrieved Series object.

# In[9]:


print('Average length: ', dfInt.Length.mean())         # same as dfInt['Length'].mean(); average length in characters
print('Standard devatiion:', dfInt.Length.std())       # average standard deviation
print('Shortest attribute name:', dfInt.Length.min())  # minimum of lengths
print('Longest attribute name:', dfInt.Length.max())   # maximum of lengths


# The average character length of the strings in this DataFrame is 8.8, with a standard deviation of 2.6.
# 
# <h3 style="color:black"> Boolean Masks</h3>
# 
# You can create **Boolean masks**, which are arrays or lists that indicate whether an element satisfies a specific condition. These masks are sometimes useful to filter a DataFrame into a smaller set of qualifying rows or columns.

# In[10]:


dfInt['isPrivateMask'] = dfInt.Methods.str.startswith('_')   # indicates private attribute names, which start with '_'
dfInt.T


# You can now apply the Boolean mask to filter rows of the DataFrame to attribute names without underscore prefixes. `~` negates each Boolean value from `True` to `False` and vice versa.

# In[11]:


aMask = ~dfInt['isPrivateMask'].values  # array of Booleans 
dfInt[aMask]


# You can also compute statistics for a particular group with the [`groupby()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) method. As an example, compute the median character length of methods for private and non-private masks.

# In[12]:


dfInt.groupby(dfInt['isPrivateMask']).median()


# You've just scratched the surface of capabilities of Pandas DataFrames. You can find a much more comprehensive documentation and examples on [pandas.pydata.org](https://pandas.pydata.org/docs/index.html). The goal here is not to learn every Pandas function and its arguments but to familiarize yourself with the tools available in this library and situations in which these tools are applicable.

# ## Slicing with `.loc`
# 
# Recall from a Python prerequisite course that slicing can also be done with [`.loc`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html) method, which takes row/column labels, a True/False array, and their ranges. Some examples are shown below.

# In[13]:


dfInt.loc[:2]  # retrieve rows with label names (not numeric indices) 0, 1, 2


# In[14]:


dfInt.loc[[1,4,6]]  # retrieve rows with label names (not numeric indices) 1, 4, 6


# In[15]:


dfInt.loc[[1,2], 'Methods']  # retrieve rows labeled 1, 2 for the column 'Methods'


# In[16]:


dfInt.loc[[1,2], [True, False, False]]  # retrieve rows labeled 1, 2 for the column 'Methods'


# ##  From Pandas to NumPy
# 
# Another operation that will be helpful in this class (and that you might remember from the prerequisite course in Python) is [`.values`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.values.html) attribute, which returns a NumPy matrix of values without row/column labels. This is a quick and convenient conversion of Pandas DataFrame to a NumPy array. 

# In[17]:


dfInt.values[:3]  # convert to numpy array and return the first 3 rows


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Optional Practice**
# 
# <span style="color:black"> Now you will practice with Pandas DataFrames.
# 
# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.

# ## Task 1
# 
# Create a DataFrame `dfA` with the column `'Name'` that contains attribute names for a Python `list` object.
# 
# <b>Hint:</b> See code above. You can use <code>dir(list)</code> to draw list attribute names.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# LsA = dir(list)
# dfA = pd.DataFrame(LsA, columns=['Name'])
# dfA.T
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# Use string's [`replace()`](https://docs.python.org/3/library/stdtypes.html#str.replace) method to replace all `'_'` characters with `''` from strings in the `Name` column and save results back to `dfA` in a new column, `'NameClean'`.
# 
# <b>Hint:</b> See code above. You can use <code>.str.replace('_','')</code> method to remove underscores in each string value stored in a DataFrame.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# dfA['NameClean'] = dfA.Name.str.replace('_','')
# dfA.T
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# Add a column, `'HasO'`, which has `True` if the name contains the letter `'o'` and `False` otherwise.
# 
# <b>Hint:</b> Try <code>str.contains('o')</code> method to calculate a Boolean value for each string in our DataFrame. See code above.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# dfA['HasO'] = dfA.NameClean.str.contains('o')
# dfA.T
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 4
# 
# Add a `Len` column with the character lengths of clean attribute names.
# 
# <b>Hint:</b> Try <code>str.len()</code> to compute length of each string stored in a DataFrame. See code above.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# dfA['Len'] = dfA.NameClean.str.len()
# dfA.T
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 5
# 
# Using `dfA`, compute the mean and standard devation for each group of attributes: those with the letter `'o'` and those without. Is there evidence that one group of attribute names is longer than the other?
# 
# <b>Hint:</b> See code above. You can use <code>std()</code> in group by to compute standard deviation of character lengths within each group.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# dfA.groupby('HasO').mean()
# dfA.groupby('HasO').std()
#     </pre>
#     </details> 
# </font>
# <hr>

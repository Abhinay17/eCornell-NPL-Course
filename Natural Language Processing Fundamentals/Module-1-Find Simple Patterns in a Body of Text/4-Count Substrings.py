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
# Review the code Professor Melnikov used to count elements of strings in the previous video. Counting is essential to many statistical, NLP, and machine learning tasks. It allows you to build distributions of elements and to quickly compare documents based on the similarity among their distributions, which you will practice later in this certificate.
# 
# ## Counting Substrings
# 
# Converting strings to **numeric representations** is essential in NLP because numeric representations can be analyzed with statistical techniques. One of the basic operations that becomes possible once strings are represented numerically is determining the **term frequency** of each substring, which is a count of the number of times a particular substring occurs. 
# 
# Use the built-in string method, [`count()`](https://docs.python.org/3/library/stdtypes.html#str.count), on the string below to count the occurences of: </span>
# 
# * the subword `'ice'` </span>
# * the number of spaces in the string</span>
# * the number of periods</span>

# In[2]:


sTxt = 'Juice, rice, and mice are iceless. Spices are priceless. '
sTxt.count('ice'), sTxt.count(' '), sTxt.count('.')  # count subwords, number of spaces, and sentences


# Note that, in this example, the number of spaces in `sTxt` is similar to the number of words in `sTxt`, whereas the number of periods in `sTxt` is similar to the number of sentences in `sTxt`.

# ## Counting Lists or List-like Elements
# 
# Python also has a convenient and fast [`Counter()`](https://docs.python.org/3/library/collections.html#collections.Counter) object in the [`collections`](https://docs.python.org/3/library/collections.html#module-collections) library that counts elements of any list-like collection. These elements can be of arbitrary data types, not just strings. `Counter()` can count all elements within the collection or specified elements and return a [dictionary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)-like object of key-value pairs, i.e., a unique element and its count. </span>
# 
# DNA sequences are expressed as long strings of characters A,C,G,T that represent nucleotides. Use `Counter()` to retrieve the frequency of each nucleotide in the DNA sequence.</span>

# In[3]:


from collections import Counter

# sequences of nucleotides A,C,T,G
sDNA = 'CTGAACTGAGACTTGGACTGAACTGACTGACTGACTGACTGCTGCTGACTGCTAGCTGGTGTGTGTGTG'
DDist = Counter(sDNA)
print(DDist)  # dictionary-like object with character distribution


# ## Application: Comparing DNA Strings
# 
# If you're working in Computational Genetics and Bioinformatics, you may want to compare two DNA sequences. One way to do this is to measure the overlap of elements in the sequences. A common approach is to find the longest common subsequence, which is a computationally intensive problem. 
# 
# When comparing many DNA sequences to one another, one way to reduce computation is to first calculate a coarse approximation of similarity by first counting the number of each nucleotide in each sequence and then comparing the distributions. After this step, slower but more precise methods can be applied to the pairs of DNA strings that have similar distributions. This approach cuts down the pairwise comparison time significantly because you do not need to compare all possible pairs.
# 
# You can look at this type of data by building a [plotly](https://plotly.com/python/bar-charts/) bar chart to express the statistics in a visually attractive manner.

# In[4]:


import plotly.express as px
fig = px.bar(y=DDist.keys(), x=DDist.values(), orientation='h')
fig.update_layout(height=100, margin=dict(t=0, b=0, l=0, r=0))


# ## Performing Operations on Strings
# 
# You can also perform operations on `Counter()` objects, which can help you integrate new documents into an analysis.

# In[5]:


print(Counter('aab') + Counter('bb'))  # add counts of corresponding elements
print(Counter('aab') - Counter('bb'))  # subtract counts of corresponding elements
print(Counter('aab') | Counter('bb'))  # union is the maximum of values
print(Counter('aab') & Counter('bb'))  # intersection is the minimum of values


# <details style="margin-top:0px;border-radius:20px"><summary>
#     <div id="button" style="background-color:#eee;padding:10px;border:1px solid black;border-radius:20px">
#        <font color=#B31B1B>▶ </font> 
#         <b>Application to NLP</b>: Continuously updating distributions
#     </div></summary>
# <div id="button_info" style="padding:10px">
# This method is useful when you have documents that you want to add to your analysis at regular intervals, $d_i$ and want to know the current distribution of, say, words in the corpus, $D_n=[d_1,d_2,...,d_n]$, when $d_{n+1}$ document needs to be added. So, instead of recomputing all words on the full corpus, $D_{n+1}$, you can count words in the new document, $d_{n+1}$, and then add these counts to the most recent counts of the corpus, $D_n$. </div> </details>
# 
# You can also access the count of a character within a string just like you would retrieve a value from a dictionary:

# In[6]:


Counter(sDNA)['A']


# The [`most_common()`](https://docs.python.org/3/library/collections.html#collections.Counter.most_common) method returns a list of tuples, where each element is a counted element followed by its count or frequency. These tuples are ordered by descending count. You can also pass an argument to `most.common()` to tell it how many elements you would like it to return.

# In[7]:


print(Counter(sDNA).most_common()) # return a list of tuples of the counted element followed by its count
TTop2 = Counter(sDNA).most_common(3)
print(TTop2) # return the tuples of the 3 most common elements followed by their counts


# You can perform slicing on this list of tuples to get the least common element and a reverse ordered list.

# In[8]:


print(Counter(sDNA).most_common()[-1])    # least common element
print(Counter(sDNA).most_common()[::-1])  # reversed ordered list


# ## Repackaging List Elements
# 
# Notice that `most_common()` returns a list of tuples, which may not be directly usable by some other functions, such as `sum()`, which counts values of a list, or `join()`, which concatenates characters from a list. In `TTop2`, the values of the lists are tuples and, hence, are not summable in a natural way. If we want to sum the counts (in the second position of each tuple), we would either have to write a custom summation, which sums counts in the second position of each tuple, or to reformat `TTop2` to extract all counts.
# 
# Either approach is suitable. 
# 
# Below is an example of a code that operates on original output from `most_common()`.

# In[9]:


print(TTop2)
print([c for c, n in TTop2])  # list comprehension to extract characters


# Now we use the [`zip()`](https://docs.python.org/3/library/functions.html#zip) function to rearrange the elements. It pulls all character letters into a tuple and all frequencies into another tuple. Both tuples are returned inside a parent list. Then we can extract either the first list of characters or the second list with counts for further processing such as summation of counts. Quickly rearranging the elements of array-like structures in this manner makes the `zip` function a powerful tool that you should make sure you understand. Here is a demonstration.

# In[10]:


print(list(zip(*TTop2)))          # group corresponding elements of each tuple
print(list(zip(*TTop2))[0])       # returns just the characters
print(list(zip(*TTop2))[1])       # returns just the counts
print(sum(list(zip(*TTop2))[1]))  # returns the sum


# Note that when the `zip()` function is not used on a list, it returns an iterator, which can be materialized or evaluated with the `list()` function. Recall from your Python experience that [`iterator`](https://wiki.python.org/moin/Iterator) is an object "ready to iterate over the elements." Iteration is (relatively) slow, so iterators allow us to postpone iteration until it is absolutely necessary. For this and other technical reasons, iterators are fast alternatives to list comprehensions and other loops, whenever applicable. The `zip()` iterator is displayed as `<zip at 0x7f8d8de88f08>`, where the number is the memory address of the iterator. In order to turn the iterator into a list, we wrap the iterator at the specified address with a `list()` function by providing an asterisk operator, `*`, which unpacks a list that follows it into individual elements. You can read more about the [unpacking operator](https://peps.python.org/pep-0448/).

# In[11]:


a, b = [1,3,5,7], [2,4,6,8] # two lists
zip(a,b)                    # zip() returns an iterator if not evaluated with list()
list(zip(a, b))             # alterantively you can zip all elements (i.e. lists) and cast to a list for viewing
list(zip(  *[ a, b ]  ))    # If a,b are packed as a list [a,b], use * to extract elements a and b


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # Optional Practice
# 
# 
# Now you will practice some of these basic counting techniques by working with the name of the bacteria *Myxococcus llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogochensis*.

# In[12]:


sTxt = 'The quick brown fox jumped Over the Big Dog And Then Jumped Again Over The Lazy Myxococcus llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogochensis'
sTxt


# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer. </span>

# ## Task 1
# 
# Split `sTxt` on spaces to form a list of words. Then assign the results to an object called `LsWord`.
# 
#  <b>Hint:</b> Use string's <code>split()</code> method with either a default or a <code>' '</code> separator argument. These both produce the same list due to lack of other whitespace characters, <code>'\t'</code>, <code>'\n'</code>, <code>'\r'</code>.

# In[13]:


LsWord = sTxt.split()


# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# LsWord = sTxt.split()
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 2
# 
# Apply the `Counter()` object to each word in the list `LsWord` to count the frequency of each word in the list. 
# 
# <b>Hint:</b> Use the <code>Counter()</code> object to count character elements in a string of characters.       

# In[ ]:


# check solution here


# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#             <pre>
# print(Counter(LsWord))
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# Use the `most_common()` method on `LsWord` to determine the frequency of each word in the list. Assign the results to a new object called `LsWord2`. Note that this is an alternate technique to what you did in Task 2. </span>
# 
# <b>Hint:</b> Use the <code>Counter()</code> object with the <code>.most_common()</code> method.

# In[ ]:


# Check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#B31B1B>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# LsWord2 = <code>Counter(LsWord)</code><code>.most_common()
# LsWord2</code>
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 4
# 
# Use the `most_common()` method on `sTxt` to determine the top five most frequently occuring characters in that string. Assign the results to a new object called `LTsnTop` and print it to the notebook to check your work. 
# 
# **Hint:** Use the <code>Counter()</code> object with the <code>.most_common()</code> method. Your answer should be:
#         <code>[(' ', 17), ('o', 11), ('l', 11), ('e', 10), ('g', 10)]</code>

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#B31B1B>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# LTsnTop = Counter(sTxt).most_common(5) # list of tuples of (str, number) elements
# LTsnTop
#             </pre>
#         </details>
# </font>
# <hr>
# 

# ## Task 5
# 
# Use the `zip()` function on `LTsnTop` rearrange elements of a list to derive a list of two tuples: one with all counted letters and another with their corresponding counts:
# 
#     [(' ', 'o', 'l', 'e', 'g'), (17, 11, 11, 10, 10)]
#     
# <b>Hint:</b> Recall that<a href="https://docs.python.org/3/library/functions.html#zip"><code>zip</code></a> takes a reference to the list of lists, not the list itself. Also, list returns an iterator, which needs to be wrapped as a list before it can be viewed.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶</font>See <b>solution</b>.</summary>
#             <pre>list(zip(*LTsnTop))   # *LTsnTop returns individual elements to zip, not a list of elements
#             </pre>
#     </details> 
# </font>

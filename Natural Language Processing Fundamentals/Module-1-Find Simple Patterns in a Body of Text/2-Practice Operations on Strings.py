#!/usr/bin/env python
# coding: utf-8

# # Setup
# 
# Reset the Python environment to clear it of any previously loaded variables, functions, and libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# Review the code Professor Melnikov used to perform operations on strings in the previous video. 
# 
# ## **Concatenating Strings**
# 
# String [concatenation](https://en.wikipedia.org/wiki/Concatenation) is one of the most fundamental string operations. Substrings can be appended with a "plus" operator (`+`) or by applying a string's [`join()`](https://docs.python.org/3/library/stdtypes.html#str.join) method to the list of substrings you would like to concatenate. 

# In[2]:


sDoc1 = 'NLP is so fun.'
sDoc2 = 'I like it a ton.'
print(sDoc1 + ' ' + sDoc2)
print(' '.join([sDoc1, sDoc2]))


# ## Standardizing Text Case
# 
# Standardizing word casing is an important preprocessing task that makes it easier to find patterns in language. The `string` methods [`lower()`](https://docs.python.org/3/library/stdtypes.html#str.lower), [`upper()`](https://docs.python.org/3/library/stdtypes.html#str.upper), [`capitalize()`](https://docs.python.org/3/library/stdtypes.html#str.capitalize), and [`title()`](https://docs.python.org/3/library/stdtypes.html#str.title) can all be helpful for this task. The `lower()` method is used most often by Python programmers.</span>

# In[3]:


print(sDoc1) # Print the original string for comparison OLEG: I added this and the following comments. Is this okay?
print(sDoc1.lower()) # Lowercase sDoc1
print(sDoc1.upper()) # Uppercase sDoc1
print(sDoc1.capitalize()) # Capitalize sDoc1
print(sDoc1.title()) # Title case sDoc1


# ## Measuring String Length
# 
# The [`len()`](https://docs.python.org/3/library/functions.html#len) function is used to measure the supplied argument. When applied to a string or a variable that contains a string, *the function returns the number of characters in that string.* Understanding the length of strings can be useful for a wide variety of NLP tasks: 
#  
# <details style="margin-top:20px;border-radius:20px">
#     <summary>
#         <div id="button" style="background-color:#eee;padding:10px;border:1px solid black;border-radius:20px">
#             <font color=#B31B1B>▶ </font> 
#             <b>Example 1</b>: Calculate Statistics of Text
#         </div>
#     </summary>
#     <div id="button_info" style="padding:10px">You can compute statistics, e.g., mean, standard deviation, mode, etc., on the lengths of words, phrases, and sentences within a large text.
#     </div>
# </details>
#  
# <details style="margin-top:10px;border-radius:20px">
#     <summary>
#         <div id="button" style="background-color:#eee;padding:10px;border:1px solid black;border-radius:20px">
#             <font color=#B31B1B>▶ </font> 
#             <b>Example 2</b>: Determine Quality of a Parsing Method
#         </div>
#     </summary>
#     <div id="button_info" style="padding:10px">Evaluating the statistical distribution of word lengths can help determine the quality of a parsing method, i.e., a method that is used to parse a document into words. If parsing returns words with an average length of 50, then it was probably parsed poorly, since the expected average word length is about 5 characters, depending on context. A medical document with many Latin, technical, and procedural terms might average 9 characters per word, but the average word length would be still be below 50.
#     </div>
# </details>
#         
# <details style="margin-top:10px;border-radius:20px"> 
#     <summary>
#         <div id="button" style="background-color:#eee;padding:10px;border:1px solid black;border-radius:20px">
#             <font color=#B31B1B>▶ </font> 
#             <b>Example 3</b>: Compare Two Documents
#         </div>
#     </summary>
#     <div id="button_info" style="padding:10px">Evaluating the similarity of two documents is often based on the number of matching words in these documents. Say, the documents are 
# 
#     sDoc1 = 'I like NLP'
#     sDoc2 = 'NLP is fun'
# 
# A brute force approach would be to compare every word in the first document to every word in the second document (assuming no duplicates), which is an expensive computational operation. This approach requires many unnecessary comparisons, because words of different lengths cannot be the same. Thus, you only need to compare words of the same length. Typically, this results in far fewer comparisons.
# </div> 
# </details>

# In[4]:


len('Keep on learning') 
len(sDoc1)


# ## Indexing Substrings
# 
#  The position of a substring can sometimes help extract meaningful content from a string. The built-in string method, [`.find()`](https://docs.python.org/3/library/stdtypes.html#str.find), searches the string for a substring marker, and returns the location of the first character in the first instance of the substring. </span>

# In[5]:


sDoc6 = 'My Phone: 123123123'
nPosIndex = sDoc6.find('Phone')   # returns position of the first character of the first instance of 'Phone' in sDoc6 string variable
print(nPosIndex)
sDoc6[(nPosIndex + 7):]           # we offset the position by 7 characters to skip 'Phone: ' (7 characters)


# Since `.find()` returns the position of only the first character in the substring, notice that in string `sDoc6`, `find('Phone')` and `find('P')` produce the same result: index 3.

# In[6]:


sDoc6.find('P')


# However, this may not be the case in a string containing more than one 'P', since the `.find()` method only returns the first character in the first instance of the given substring. In `sDoc7` below, substring 'P' occurs in both 'Poor' and 'Phone'; however, only the index for the first instance of 'P' is returned. 

# In[7]:


sDoc7 = 'Poor Phone Quality'
sDoc7.find('P')


# To find the location of the word 'Phone' in a string like `sDoc7`, you would need to specify the entire substring 'Phone', rather than just 'P'.

# In[8]:


sDoc7.find('Phone')


# You can also test whether a substring is contained within a string. The [`.startswith()`](https://docs.python.org/3/library/stdtypes.html#str.startswith) method checks whether the string begins with the substring of interest and returns this result as `true` or `false`. Similarly,  [`.endswith()`](https://docs.python.org/3/library/stdtypes.html#str.endswith) checks whether the string ends with the substring.</span>

# In[9]:


'Phone' in sDoc6, sDoc6.startswith('My'), sDoc6.endswith('My')  # returns a tuple with results from each operation


# ## Replacing Whitespace Characters
# 
# The [`replace()`](https://docs.python.org/3/library/stdtypes.html#str.replace) method is an important approach you can use to clean and standardize strings. For example, invisible characters such as `\n` and `\r` are usually changed to whitespace as part of text standardization.</span>

# In[10]:


'NLP\nis gr8!'.replace('\n',' ').replace('gr8', 'great')   # returns a cleaner string that is easier to read


# ## Checking a String's State
# 
# Often, you'll want to check the state of a given string to see what needs to be standardized. There are many methods for doing this, including [<code>isupper()</code>](https://docs.python.org/3/library/stdtypes.html#str.isupper),
# [<code>islower()</code>](https://docs.python.org/3/library/stdtypes.html#str.islower),[<code>isdecimal()</code>](https://docs.python.org/3/library/stdtypes.html#str.isdecimal), [<code>isalpha()</code>](https://docs.python.org/3/library/stdtypes.html#str.isalpha), [<code>isalnum()</code>](https://docs.python.org/3/library/stdtypes.html#str.isalnum), and more. These methods return a boolean value, either `True` or `False` depending on whether the string matches the specifications of the method.

# In[11]:


'Yesssss!'.isupper(), 'Oh, Noooo'.lower().islower()    # returns a tuple with results from each operation


# ## Stripping Whitespace
# 
# Preprocessing a string usually involves removing unnecessary leading and trailing spaces. The [<code>strip()</code>](https://docs.python.org/3/library/stdtypes.html#str.strip) method removes these extra spaces.

# In[12]:


('     a space-padded string     ').strip()    # strips left and right space characters


# ## Splitting a String
# 
# The [<code>split()</code>](https://docs.python.org/3/library/stdtypes.html#str.split) method breaks a string at whitespace characters, which include space (<code>' '</code>), Tab (<code>\t</code>), newline (<code>\n</code>), or carriage return (<code>\r</code>).</span> 

# In[13]:


print('split on whitespace characters: tab \t, newline \n, carriage return \r, and space'.split())


# You can also split a string at a specified character. 

# In[14]:


print('\t\r\n \t\r\n \t\r\n '.split('\n'))
print('\t\r\n \t\r\n \t\r\n '.split(' '))
print('\t\r\n \t\r\n \t\r\n '.split('\n '))


# Note that spaces remain in substrings if we split on <code>\n</code>.

# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # Optional Practice
# 
# Now you will practice techniques to perform operations on strings. To start, run the following code to load and print the same string you worked with in the last exercise: 

# In[15]:


sTxt = 'The quick brown fox jumped Over the Big Dog And Then Jumped Again Over The Lazy Myxococcus llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogochensis'
sTxt


# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.
# 
# ## Task 1
# 
# Convert <code>sTxt</code> to lowercase, save the output as <code>sTxt1</code>, and print <code>sTxt1</code>.
# 
# **Hint:** Try calling <code>.lower()</code> method on <code>sTxt</code>.

# In[19]:


sTxt1 = sTxt.lower()
print(sTxt1)


# <font color=606366>
#     <details><summary><font color=B31B1B>▶ </font>See <b>solution</b>.</summary>
# <code>sTxt1 = sTxt.lower()
# print(sTxt1)</code></p>
#         </details>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# Replace all instances of `"dog"` in the <code>sTxt1</code> string with `"fox"`. Save the output to a new object called <code>sTxt2</code>.
# 
# <b>Hint:</b> Try using the <code>str.replace()</code> method.

# In[20]:


sIn, sOut = 'dog', 'fox'
sTxt2 = sTxt1.replace(sIn, sOut)


# <font color=606366>
#     <details><summary><font color=B31B1B>▶ </font>See <b>solution</b>.</summary>       
# <p><code>sIn, sOut = 'dog', 'fox'
# sTxt2 = sTxt1.replace(sIn, sOut)
# sTxt2.find(sIn)     # confirm 'dog' is not in the string. 
#                     # Output of -1 means search string not found
# sTxt2</code>
#         </details>
# </font>
# <hr>

# ## Task 3
# 
# Split <code>sTxt2</code> on spaces, save the output as <code>sTxt3</code>, and print <code>sTxt3</code>.
# 
# <b>Hint:</b> The function <code>str.split()</code> uses whitespace as the default separator character, not a space. Check out the documentation for this method. Alternatively, you could use the more laborious method of creating a loop.

# In[21]:


sTxt3=sTxt2.split(' ')
print(sTxt3)


# <font color=606366>
#     <details><summary><font color=B31B1B>▶ </font>See <b>solution</b>.</summary>
#         <pre>
# <code>sTxt3 = sTxt2.split(' ')
# print(sTxt3)</code>
#             </pre>
#     </details> 
# </font>
# 
# <hr>

# ## Task 4
# 
# Join <code>sTxt3</code> back together, but this time separate the substrings from one another with a dash (`'-'`). Save the results to <code>sTxt4</code>, and print <code>sTxt4</code>. 
# 
# <b>Hint:</b> Try using the <code>str.join()</code> method. See documentation, if needed.

# In[22]:


sTxt4 = '-'.join(sTxt3)
print(sTxt4)


# <font color=606366>
#     <details><summary><font color=B31B1B>▶ </font>See <b>solution</b>.</summary>
#         <pre>
# <code>sTxt4 = '-'.join(sTxt3)
# print(sTxt4)</code>
#             </pre>
#         </details>
# </font>
# 
# <hr>

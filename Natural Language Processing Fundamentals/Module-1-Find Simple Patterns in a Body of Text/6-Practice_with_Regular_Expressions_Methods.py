#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.  

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import nltk, re


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# Review some of the simple but powerful regex techniques that Professor Melnikov demonstrated in the previous video. In particular, you will examine several different methods of the [`re`](https://docs.python.org/3/library/re.html#module-re) library, including: 
# 
# * [`search()`](https://docs.python.org/3/library/re.html#re.search) 
# * [`findall()`](https://docs.python.org/3/library/re.html#re.findall)
# * [`sub()`](https://docs.python.org/3/library/re.html#re.sub)
# 
# You'll also practice using some of the more complex search patterns Professor Melnikov used in the video. Start by loading and printing `sFix`, the string you will be working with in this coding activity.

# In[2]:


sFix = 'fix is in prefix, suffix, affix, and fixture'
print(sFix)


# ## First Match
# 
# You can use the `re.search()` method to find only the first match of a pattern in a given string. Instead of returning a modified string, this search returns a [`re.Match`](https://docs.python.org/3/library/re.html#match-objects) object, which provides information about the first match, including its location within the string. 
# 
# The example below looks for words starting with `'fix'` followed by at least one word character. Whenever a non-word character is reached (such as a space), the search stops.  The search reports that the first match is the word `'fixture'`, which starts in position 37 and ends in position 44 (which is the length of string `s`). 

# In[3]:


m = re.search(pattern='fix\w+', string=sFix, flags=0)   # match "fix" followed by at least one word character
m

print('regex object is \t\t', re.search('fix\w+', sFix))
print('found string is \t\t', re.search('fix\w+', sFix)[0])
print('starting index position\t', re.search('fix\w+', sFix).start())
print('ending index position\t', re.search('fix\w+', sFix).end())


# As you can see, the  `re.Match` object includes the string `'fixture'`, which matches the specified search pattern `'fix\w+'`.  You can access this string by slicing the `re.Match` object directly or calling its [`.group()`](https://docs.python.org/3/library/re.html#re.Match.group) method; both methods return the same result.

# In[4]:


m[0], m.group(0)


# The meta information the `re.Match` object contains also includes the starting and ending index positions.

# In[5]:


print('starting index position\t', m.start())
print('ending index position\t', m.end())


# <details style="margin-top:0px;border-radius:20px"><summary>
#     <div id="button" style="background-color:#eee;padding:10px;border:1px solid black;border-radius:20px">
#        <font color=#B31B1B>▶ </font> 
#         <b>Application in NLP</b>: Finding User IDs
#     </div></summary>
# <div id="button_info" style="padding:10px">This information can be used to find the location of some trigger word and then look for information related to the trigger word. For example, we could look for <code>'SkypeID:'</code> trigger and then extract the ID itself, which is likely to follow such a trigger. If no substring is found, the search returns <code>None</code>, which is evaluted as <code>False</code>.</div> </details>
# 
# ## All Matches
# 
# 
# To retrieve all instances of matched substrings, you can use the [`re.findall()`](https://docs.python.org/3/library/re.html#re.findall) method, which returns a list of strings without meta data. This can be useful if you want to know the presence of string patterns, or their counts, in a document, but do not need their locations. The following code finds all matched substrings that contain a digit followed by two zeros.

# In[6]:


s01 = "1110001110001"
re.findall(pattern='\d00', string=s01)


# Match any symbol in character class `[]` at least once:

# In[7]:


re.findall(pattern='[nNlLpP]+', string='We ❤ NLP')


# Match any strings separated by the pipe operator, using the `re.IGNORECASE` flag to ignore the letter casing:

# In[8]:


re.findall(pattern='we|nlp', string='We ❤ NLP', flags=re.IGNORECASE)


# Match the all instances of the string run, regardless of case or whether it is followed by word characters: 

# In[9]:


sRun = 'I am a runner. I enjoy running at my morning Runs'
re.findall(pattern='run\w*', string=sRun, flags=re.IGNORECASE)


# ## All Matches with Metadata
# 
# If search meta information is valuable, we can use [`re.finditer()`](https://docs.python.org/3/library/re.html#re.finditer), which returns an [iterator](https://docs.python.org/3/glossary.html#term-iterator) of match objects. In order to view the results of the iterator, we should apply the `list()` function to it.

# In[10]:


LmoResults = list(re.finditer('fix', sFix))
LmoResults


# Now, we can show all positions of matched substrings.

# In[11]:


[mo.start() for mo in LmoResults]


# ## Substitution
# 
# The `re.sub()` method allows a pattern in a string to be replaced.

# In[12]:


re.sub(pattern='NLP', repl='NLP & Python', string='We ❤ NLP')


# The real power of regex comes from the ability to combine many rules into one pattern, such as this:

# In[13]:


sDNA = 'ACGTAGCTACGTATGACGTAACGT'
re.sub(pattern='A[AT]|[GT]A', repl='--', string=sDNA)


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # Optional Practice
# 
# Now you will practice some basic regex patterns that were introduced in the previous video. To start, load a famous quote by U.S. President Franklin.

# In[14]:


sQuote = '`Tell me and I forget, teach me and I may remember, involve me and I learn.` Benjamin Franklin'
sQuote


# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.

# ## Task 1
# 
# Return the match object for the first instance of the `'me'` string.
# 
# <b>Hint:</b> Try the <code>re.search()</code> method.

# In[15]:


re.search('me', sQuote)


# 
# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# re.search('me', sQuote)    # return match object for the first match of "me"
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 2
# 
# Return a list of all words in `sQuote` containing the substring `'me'`. One such word is `'remember'`.
# 
# <b>Hint:</b> Try <code>re.findall()</code> method. The regex pattern should allow for any number of the leading and trailing word characters, <code>\w</code>. You can use <code>*</code> to allow 0 or more repetitions of these.

# In[16]:


re.findall('\w*me\w*', sQuote)


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1>▶</font>
#             See <b>solution</b>.
#         </summary>
#             <pre>
# re.findall('\w*me\w*', sQuote)          # all words containing "me"
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 3
# 
# Return the starting position of the third word that contains the substring `"me"`.
# 
# <b>Hint:</b> Try <code>re.finditer()</code>, which needs to be wrapped into a list. Then you can slice it at the 3rd element (with index 2) and call the <code>start()</code> method for the starting position.

# In[17]:


list(re.finditer('\w*me\w*', sQuote))[2].start()


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1>▶</font>
#             See <b>solution</b>.</summary>
#             <pre>
# list(re.finditer('\w*me\w*', sQuote))[2].start()   # starting position of the 3rd word(containing "me")
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 4
# 
# Return a list of match objects of all words starting with `"I"` or `"i"`.
# 
# <b>Hint:</b> You need to establish a leading word boundary with <code>'\b'</code> and trailing arbitrary word characters with <code>'\w*'</code>. You need to force regex pattern string to be <a src=https://www.python.org/dev/peps/pep-0498/>raw</a> because you're using <code>'\b'</code>. This is not the case, when you use the special characters <code>'\n'</code>, <code>'\r'</code>, or <code>'\t'</code>, which Python interprets as such without an explicit raw string indicator.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1>▶</font>
#             See <b>solution</b>.
#         </summary>
#             Here are a couple of ways to accomplish this.
#             <pre>
# # A list of match objects of all words starting with "I" or "i"
# list(re.finditer(r'\bi\w*', sQuote, flags=re.IGNORECASE)) 
# list(re.finditer(r'\b[iI]\w*', sQuote))
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 5
# 
# Return the total count of characters of all words starting with `"I"` or `"i"`.
# 
# <b>Hint:</b> You can iterate over the results of <code>re.finditer()</code> (even without wrapping it as a list). When doing so, compute the difference between ending character and starting character + 1.

# In[18]:


sum([mo.end() - mo.start() for mo in re.finditer(r'\b[iI]\w*', sQuote)]) 


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1>▶</font>See <b>solution</b>.</summary>
#             <pre>
# # Total count of characters of all words starting with `"I"` or `"i"`
# sum([mo.end() - mo.start() for mo in re.finditer(r'\b[iI]\w*', sQuote)])  # solution 1
# 
# len("".join(re.findall(r'\b[Ii]\w*', sQuote)))                            # solution 2
# 
# sum([len(i) for i in re.findall(r'\b[iI]\w*', sQuote)])                   # solution 3
#             </pre>
#         </details>
# </font>
# <hr>

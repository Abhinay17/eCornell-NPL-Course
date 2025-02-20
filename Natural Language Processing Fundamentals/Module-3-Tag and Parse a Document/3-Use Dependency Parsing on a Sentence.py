#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Clear the Python environment of any previously loaded variables, functions, and libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import spacy, pandas as pd, nltk
from spacy import displacy

print('SpaCy version:', spacy.__version__) # SpaCy>=3 uses en_core_web_sm pretrained NLP model


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# ## Dependency Parsing
# 
# In this notebook, you will practice how to build a multi-level tree that displays the token dependencies in a sentence. Begin by wrapping the sentence with SpaCy's `nlp` and store this object in `doc`.

# In[2]:


sNews = 'An independent newspaper, The Cornell Daily Sun, was founded by William Ballard Hoyt in 1880.'
nlp = spacy.load('en_core_web_sm')   # text-processing pipeline object for English
doc = nlp(sNews)                     # A sequence of Token objects
doc


# The resulting object contains useful attributes, some of which can be used to identify relationships between parent and child words. Loop through the word tokens within the  object and print out some of their attributes.

# In[3]:


LTsTags = [(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_) for token in doc]
LTsTags


# Hierarchical relationships can be easier to interpret when visualized in tabular format using a Pandas DataFrame. The `Dep` row indicates the `'founded'` token as the `ROOT` of this tree. 

# In[4]:


LsCols = ['Token','Tag','Dep','Head','Head_Tag']
pd.DataFrame(LTsTags, columns=LsCols).T


# Note that the dependency relation in the `Dep` row uses the `'founded'` token as the `ROOT` of the dependency tree. The dependency parsing algorithm is not perfect, and some sentences can have multiple or no roots.
# 
# You can also view the object as a tree with the `SpaCy` library's `displacy` visualizer. 

# In[5]:


displacy.render(doc, style='dep', jupyter=True, options={'distance': 90, 'compact':True, 'bg':'lightgray'})


# <span style="color:black"> The arrows point from a parent to its child. Modifiers may be located further in a sentence than the word they modify. Notice that there is an arrow (a.k.a. `"edge"`) from `"founded"` (parent) to `"newspaper"` (child), which are separated by at least five tokens.

# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Optional Practice**
# 
# Now practice implementing the above concepts.

# In[ ]:


sQuote = "If you can't explain it simply, you don't understand it well enough." # quote from Albert Einstein
sQuote


# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.
# 
# ## Task 1
# 
# <span style="color:black"> Transform `sQuote` into a SpaCy `nlp` object, `docQ`. Wrap its dependency tree attributes (same as those in the Review section) into a dataframe, `dfQ`. Print the transposed dataframe and examine the table to determine the root word(s).
#     
# <b>Hint:</b> Follow the dataframe example above.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# docQ = nlp(sQuote)
# dfQ = pd.DataFrame([(t.text, t.tag_, t.dep_, t.head.text, t.head.tag_) for t in docQ], columns=LsCols)
# dfQ.T   # the root is "understand"
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# Build a tree diagram using SpaCy's `displacy` visualizer and find the tree root(s).
# 
# <b>Hint:</b> Follow the displacy example above.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# displacy.render(docQ, style='dep', jupyter=True, options={'distance': 90, 'compact':True, 'bg':'lightgray'})
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# Use list comprehension to iterate over tokens of `docQ` object to find the first root. Save this string in `sRoot` variable.  Try it on `docQ`. Save the root as `sRoot` string variable.
# 
# <b>Hint:</b> In the loop comprehension above (executed on <code>docQ</code> variable), you need to add a condition for the <code>token.dep_</code> field.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# sRoot = [t.text for t in docQ if t.dep_=='ROOT'][0]  # root of the tree
# sRoot
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 4
# 
# Use list comprehension to identify the children of `sRoot` and save this list as `LsRootChildren`.
# 
# <b>Hint:</b> This is similar to previous task, but the condition is on <code>token.head.text==sRoot</code>.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# LsRootChildren = [t.text for t in docQ if t.head.text==sRoot]  # all children of ROOT
# print(LsRootChildren)
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 5
# 
# Use `nltk.corpus.inaugural.raw()` to download Roosevelt's 1945 (`'1945-Roosevelt.txt'`) inaugural speech as a raw string. Perform string preprocessing by replacing any newline character with a space, any double-dash with a space, and any double-space with a single space. Save the resulting string as `sPres` and print.
# 
# <b>Hint:</b> Try <code>nltk.corpus.inaugural.raw()</code> method.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# import nltk
# _ = nltk.download(['inaugural'], quiet=True)
# sPres = nltk.corpus.inaugural.raw('1945-Roosevelt.txt').replace('\n',' ').replace('--',' ').replace('  ',' ')
# print(sPres)
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 6
# 
# Wrap dependency tree attributes (same as those in the Review section) for `sPres` into the dataframe, `dfPres`. It should have 5 columns and 623 rows.
# 
# <b>Hint:</b> Follow the dataframe example above.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# docPres = nlp(sPres)
# dfPres = pd.DataFrame([(t.text, t.tag_, t.dep_, t.head.text, t.head.tag_) for t in docPres], columns=LsCols)
# dfPres.T
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 7
# 
# The `dfPres` dataframe shows multiple roots (approximately one from each sentence). Identify the most frequent root.
# 
# <b>Hint:</b> Try filtering your dataframe on <code>Dep</code> column being equal to <code>'ROOT'</code>. Then use group by to count tokens. You could also use <code>collections.Counter()</code>

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# dfPres[dfPres.Dep=='ROOT'].groupby('Token').count().sort_values('Tag', ascending=False).head()
#     </pre>
#     </details> 
# </font>
# <hr>

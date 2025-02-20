#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.

# In[ ]:


get_ipython().run_line_magic('reset', '-f')
# !pip -q install svgling >log
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import nltk, svgling


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# Below we use `nltk.CFG()` to build a context-free grammar ([CFG](https://en.wikipedia.org/wiki/Context-free_grammar)) from a string. Grammar is a set of language production rules. Thus, the rule `S -> NP VP` indicates that a sentence is a combination of a noun phrase, `NP`, and a verb phrase, `VP`, each of which is also recursively defined in this grammar. For example, a verb phrase is either a verb, `V`, and a noun phrase, i.e. `V NP`, or verb phrase and prepositional phrase, `PP`, which in turn is defined as `PP -> P NP`. 

# In[ ]:


groucho_grammar = nltk.CFG.fromstring("""
    S -> NP VP
    PP -> P NP
    NP -> Det N | Det N PP | 'I'
    VP -> V NP | VP PP
    Det -> 'an' | 'my'
    N -> 'elephant' | 'pajamas'
    V -> 'shot'
    P -> 'in' """)      # more info: ? nltk.CFG

type(groucho_grammar)


# Next we split a sentence into words, initialize the so-called chart parser, and parse our list of these words. This builds a list of possible constituency parsing trees. Each tree is an interpretation of the sentence using the specified grammar.

# In[ ]:


LsWords = 'I shot an elephant in my pajamas'.split()   # See example at http://www.nltk.org/book/ch08-extras.html

LTrees = list(nltk.ChartParser(groucho_grammar).parse(LsWords))  # wrap a generator as a list
print(f'Number of trees: {len(LTrees)}')
print(LTrees)


# We can print each tree with a bit more structure. Groucho grammar results in two trees, which rise from the ambiguity in parsing this sentence. In particular, it's not clear whether the meaning is 
# 1. `'[I] in my pajamas'` or 
# 1. `'an elephant in my pajamas'`

# In[ ]:


for tree in LTrees:  # view more info: ? nltk.ChartParser
    print(tree)
    
trees = [tree for tree in LTrees]

print(f"Tree count: {len(LTrees)}")


# Next we can visualize each top-bottom tree using the `svgling` library.
# 
# In the first tree, the sentence `S` relates `I` to each 3rd level branch. So, we deduce `I shot an elephant` and `I in my pajamas`.

# In[ ]:


svgling.draw_tree(trees[0])


# In the second tree, a 3rd level `NP` relates an `elephant` to `in my pajamas`. So, we deduce `an elephant in my pajamas`.

# In[ ]:


svgling.draw_tree(trees[1])


# NLTK also provides large grammars with "production" rules that are tied even to individual words. Here is an example of this type of large grammar.

# In[ ]:


_ = nltk.download(['large_grammars'], quiet=True)
grammar  = nltk.data.load('grammars/large_grammars/atis.cfg')
print(str(grammar)[:510])


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Optional Practice**
# 
# In these practice tasks, you will practice using constituency parsing on simple sentences, and visualizing your results as trees. 
# 
# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.

# In[ ]:


LsWords3 = 'I saw an elephant in my pajamas'.split()   # See example at http://www.nltk.org/book/ch08-extras.html
LsWords4 = 'We saw a cat in my boots'.split()


# ## Task 1
# 
# Modify `groucho_grammar` to correctly parse `LsWords3` above, where `'shot'` is replaced with `'saw'`. Save the new grammar as `groucho_grammar3`. Print all constituency trees on screen.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# groucho_grammar3 = nltk.CFG.fromstring("""
#     S -> NP VP
#     PP -> P NP
#     NP -> Det N | Det N PP | 'I'
#     VP -> V NP | VP PP
#     Det -> 'an' | 'my'
#     N -> 'elephant' | 'pajamas'
#     V -> 'saw'
#     P -> 'in' """)      # more info: ? nltk.CFG
# 
# LTrees3 = list(nltk.ChartParser(groucho_grammar3).parse(LsWords3))  # wrap a generator as a list
# _ = [print(t) for t in LTrees3]
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# Modify `groucho_grammar` to correctly parse `LsWords4` above. Save the new grammar as `groucho_grammar4`. Print all constituency trees on screen.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# groucho_grammar4 = nltk.CFG.fromstring('''
#     S -> NP VP
#     PP -> P NP
#     NP -> Det N | Det N PP | 'We'
#     VP -> V NP | VP PP
#     Det -> 'a' | 'my'
#     N -> 'cat' | 'boots'
#     V -> 'saw'
#     P -> 'in' ''')      # more info: ? nltk.CFG
# 
# LTrees4 = list(nltk.ChartParser(groucho_grammar4).parse(LsWords4))  # wrap a generator as a list
# _ = [print(t) for t in LTrees4]
#     </pre>
#     </details> 
# </font>
# <hr>

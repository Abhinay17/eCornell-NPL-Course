#!/usr/bin/env python
# coding: utf-8

# # **Review**
#  
# Clear the Python environment of any previously loaded variables, functions, and libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video. 
# 
# SpaCy library uses a [`en_core_web_sm`](https://spacy.io/models/en#en_core_web_sm), a small set of pre-processing functions for English text (pre-trained on WWW content).

# In[1]:


get_ipython().run_line_magic('reset', '-f')
# !pip -q install spacy>=3.0.0 > log               # quietly upgrade SpaCy, which uses `en_core_web_sm` model
# !python -m spacy download en_core_web_sm > log   # download small English model

from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import nltk, spacy, pandas as pd
from spacy import displacy
print('SpaCy version:', spacy.__version__)


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# In this notebook you will use chunking to shallow parse sentences.
# 
# ## Chunking
# 
# [Chunking](https://www.nltk.org/book_1ed/ch07.html) is a type of **partial** or **shallow** parsing that retrieves only flat, non-overlapping segments from a sentence rather than a hierarchically parsed representation. A simple bracketing notation can be used to indicate the start, end, and type of each chunk. 
# 
# ## Using Regex Notation to Chunk a Sentence
# 
# You can use regex rules to tell Python how to use POS tags to parse a sentence by defining rules for different types of phrases. For example: 
# 
# 1. The code `"VP: {<V.*>+}"` indicates a verb phrase (marked as `VP`), which must have at least one `<V.*>` tag (indicated by a `+`). 
#   * `V.` allows any 2-letter tags starting with `V`, such as `VB` (base form) and `VP` (verb phrase).
#   * `V.*` allows at least 2 letter tags starting with `V`, such as `VBD` (past tense), `VBG` (gerund/present participle), `VBN` (past participle), `VBP` and `VBZ`. 
# 1. `"V2V: {<V.*> <TO> <V.*>}"` is a chunk we named `V2V` which must start with a verb in any form, followed by `"to"`, followed by another verb in any form. For example:
# 
#   $$\newcommand{\u}[2]{\underset{\text{#1}}{\,\text{ #2 }\,}} "\u{PRP}{We}\underbrace{\u{VBP}{dress}\u{TO}{to}\u{VB}{impress}}_{\text{V2V}}."\,\,\,\,\,,\,\,\,\,\, "\u{PRP}{He}\underbrace{\u{VBD}{loved}\u{TO}{to}\u{VB}{work}}_{\text{V2V}}."$$
#   
# 1. The code `"NP: {<JJ>?<NN.*>}"` finds noun phrases (marked as `NP`), which may start with a single adjective (identified with a tag `JJ`) and are followed by any noun with a tag `NN.*`, which includes `NNS`, `NNP`, and `NNPS`. For example, this sentence has several noun phrases matching the regex pattern:
# 
# $$"\underbrace{\u{NNP}{Alice}}_{\text{NP}} \u{VBD}{had}\u{DT}{a} \underbrace{\u{NN}{dog}}_{\text{NP}},\u{DT}{a} \underbrace{\u{JJ}{brown}\u{NN}{fox}}_{\text{NP}},\u{CC}{and}\underbrace{\u{JJ}{many}\u{NNS}{cats}}_{\text{NP}}."$$
# 
# 

# ## Identify Chunks with a Pre-Determined Grammar
# 
# To identify chunks with pre-determined grammar, we first need to identify POS tags for all elements of a sentence.

# In[2]:


sNews = 'An independent newspaper, The Cornell Daily Sun, was founded by William Ballard Hoyt in 1880.'
LsNewsTokens = nltk.word_tokenize(sNews)   # parse a sentence into a list of word tokens
LTsNewsTags = nltk.pos_tag(LsNewsTokens)   # POS-tag all word tokens
print(LTsNewsTags)


# The following noun phrase looks for the pattern `determiner + adjective + noun`. 
# 
# 1. The `'?'` allows at most one determiner. 
# 1. The `'<JJ>*'` allows none or more adjectives, which are identified by POS tag `'JJ'`
# 1. The `'<NN.*>'` allows different nouns (`'NN'`, `'NNP'`, etc.), containing 2 or more letters (starting with `'NN'`) in their POS tag.
# 
# Use this pattern to perform shallow parsing on the sentence above.

# In[3]:


sNounPhraseGrammar = 'NP: {<DT>?<JJ>*<NN.*>*}' # determiner + adjective + noun
cp = nltk.RegexpParser(sNounPhraseGrammar)     # initialize the parser with a grammar definition
treeNews = cp.parse(LTsNewsTags)               # parse a sentence (words+tags) into a shallow tree of words and chunks
print(treeNews)


# The algorithm correctly identified the name of the paper and its founder. The output is presented in the form of a schematic tree diagram, where `'S'` indicates the full sentence (as a root of the tree). Chunking is a type of shallow parsing, because this tree has a single level below its root. The root is linked to leaves (in their original order), some of which are individual word tokens and others are chunks. The chunk is made up of leaves that matched the grammar pattern.
# 
# ## Working with the Parsed Tree
# 
# The parsed tree is cumbersome to work with programmatically. So, if you need a list of identified chunks, you can iterate over tree leaves and collect only those that do not have a label `'S'`, which indicates that the leaf is an individual word token and is not a chunk phrase. Investigate the `Tree2Chunks()` function below, which prints only the chunk phrases and their tags (=labels).

# In[4]:


def Tree2Chunks(tree):
    JoinLeaves = lambda tree: (' '.join(leaf[0] for leaf in tree.leaves()), tree.label()) # chunk phrase and its label
    return [JoinLeaves(tree) for tree in tree.subtrees() if tree.label()!='S']
print(Tree2Chunks(treeNews)) 


# ## Chunks with Larger Grammar
# 
# The following (somewhat sophisticated) chunking pattern searches for different types of grammatical phrases.
# It leaves out `"The"` determiner from the newspaper's name by design. All grammars are labeled with names that are allowed to repeat. So, verb phrases are identified by `'VP'`. Noun phrases are identified with `'NP'`. 
# 
# The basic grammar syntax is as following:
# 
# 1. Each chunk phrase definition:
#     1. has a tag(=label), a colon, and a definition wrapped in curly brackets. E.g. `'VP: {...}'`
#         1. different definitions can have the same tags (for example, there are several combinations that qualify as a verb phrase.)
#     1. contains Penn Tree POS tags in angle brackets. E.g. `'<NN>'`
#     1. respects regex syntax
# 1. A grammar is a collection of newline-separated phrase definitions (`'VP:{...}\nNP:{...}'`)
# 
# Take some time to investigate chunk definitions. You can try this with your own sentences or chunks.
# 
# 

# In[5]:


sGrammar=r"""

    # Verb Phrase Definitions
    
    VP: {<V.*>+}
    VP: {<ADJ_SIM><V_PRS>}
    VP: {<ADJ_INO><V.*>}
    VP: {<V_PRS><N_SING><V_SUB>}
    VP: {<N_SING><V_.*>}
    
    
    # Noun Phrase Definitions

    NP: {<N.*><PRO>}
    NP: {<ADJ.*>?<N.*>+ <ADJ.*>?}
    NP: {<N_SING><ADJ.*><N_SING>}
    
    
    # Determiner followed by a NP (Noun Phrase, as defined above)

    DNP: {<DET><NP>}
    
    
    # EDIT -- THIS NEEDS TO BE DEFINED (Oleg?)
    
    PP: {<P>*}
    PP: {<P><N_SING>}
    PP: {<ADJ_CMPR><P>}
    PP: {<ADJ_SIM><P>}
    
    
    # Noun Phrase followed by a DNP (Determiner followed by a Noun Phrase, as defined above)
    
    DDNP: {<NP><DNP>}
    
    
    # EDIT -- PP NEEDS TO BE DEFINED (Oleg?) and then this finished out
    
    NPP: {<PP><NP>+}
    """
cp = nltk.RegexpParser(sGrammar)
treeNews2 = cp.parse(LTsNewsTags)
print(treeNews2)
print(Tree2Chunks(treeNews2))


# ## Chunking with SpaCy
# 
# [SpaCy](https://spacy.io/usage/models#quickstart) is a powerful NLP package, which uses sophisticated pre-trained models to provide rich meta information about sentences and documents. Here we load one small (`'sm'`) English (`'en'`) model, which is trained on web content. A string sentence is then wrapped into an `nlp` object, which provides access to underlying text, noun phrases, labels, root noun of each phrase, and much more. SpaCy provides superior quality parsers and labels. Its only drawback is that it's still slower than comparable NLTK tools. 

# In[6]:


nlp = spacy.load("en_core_web_sm") # text-processing pipeline object for English
doc = nlp(sNews)                   # A sequence of Token objects
LTsNP = [(c.text, c.label_, c.root.text) for c in doc.noun_chunks]
pd.DataFrame(LTsNP, columns=['Noun phrase', 'label', 'Root noun'])


# SpaCy also has a [displacy](https://spacy.io/usage/visualizers) visualizer object used to render noun phrases in color, with tags. Below is our sentence, where organization, person, and date are identified. It is amazing that SpaCy was able to "understand" the role of these phrases without us providing any additional definitions or context. Perhaps, SpaCy’s model has already "seen" these phrases while training on the web corpus of documents.

# In[7]:


spacy.displacy.render(doc, style='ent', jupyter=True)


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Optional Practice**
# 
# Now you will practice on a few related tasks.
#     
# If you need to recall the definitions of Penn Tree tags, use this code:
# 
#     pd.set_option('max_rows', 100, 'display.max_colwidth', 0)
#     DTagSet = nltk.data.load('help/tagsets/upenn_tagset.pickle')  # dictionary of POS tags
#     pd.DataFrame(DTagSet, index=['Definition', 'Examples']).T.sort_index().reset_index().rename(columns={'index':'Tag'})
#     
# 

# In[ ]:


# Source: https://en.wikipedia.org/wiki/Python_(programming_language)
sPython = "Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands as a successor to ABC programming language, which was inspired by SETL,capable of exception handling and interfacing with the Amoeba operating system. Its implementation began in December 1989. Van Rossum shouldered sole responsibility for the project, as the lead developer, until 12 July 2018, when he announced his \"permanent vacation\" from his responsibilities as Python's Benevolent Dictator For Life, a title the Python community bestowed upon him to reflect his long-term commitment as the project's chief decision-maker. In January 2019, active Python core developers elected a 5-member \"Steering Council\" to lead the project. As of 2021, the current members of this council are Barry Warsaw, Brett Cannon, Carol Willing, Thomas Wouters, and Pablo Galindo Salgado." 


# In[ ]:


LTsPythonTags = nltk.pos_tag(nltk.word_tokenize(sPython))   # POS-tag all word tokens
print(LTsPythonTags)


# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.

# ## Task 1
# 
# In `sPython` find noun phrases which start with one determiner, followed by one adjective, followed by one noun of any type. For example, your grammar should identify the following chunks:
# 
#     [('the late 1980s', 'NP'), ('the current members', 'NP')]
#     
# <b>Hint:</b> Try a grammar <code>'NP: {&lt;DT&gt;&lt;JJ&gt;&lt;NN.*&gt;}'</code>

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# cp = nltk.RegexpParser('NP: {&lt;DT&gt;&lt;JJ&gt;&lt;NN.*&gt;}')
# print(Tree2Chunks(cp.parse(LTsPythonTags)))
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# In `sPython` find number phrases which start with `IN`-tagged word (preposition or subordinating conjunction), followed by a proper singular noun, followed by a numeral or cardinal. For example, your grammar should find the following chunks labeled as `Date`.
# 
#     [('in December 1989', 'Date'), ('In January 2019', 'Date')]
#     
#  <b>Hint:</b> Try a grammar <code>'Date: {&lt;IN&gt;&lt;NNP&gt;&lt;CD&gt;}'</code>

# In[ ]:


# check solution here


# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# cp = nltk.RegexpParser('Date: {&lt;IN&gt;&lt;NNP&gt;&lt;CD&gt;}')
# print(Tree2Chunks(cp.parse(LTsPythonTags)))
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# In `sPython` find entity phrases, which have two or more nouns in any form (`NN`, `NNP`, `NNS`, ...). For example, your grammar should retieve the following chunks labeled as `Entity`:
# 
#     [('Guido van Rossum', 'Entity'), ('Centrum Wiskunde', 'Entity'), , ('ABC programming language', 'Entity'), ...]
#     
#  <b>Hint:</b> You can use two nouns with the third one as optional.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# # Solution 1:
# cp = nltk.RegexpParser('Entity:{&lt;NN.*>&lt;NN.*>&lt;NN.*>*}')
# print(Tree2Chunks(cp.parse(LTsPythonTags)))
# # Solution 2:
# cp = nltk.RegexpParser('Entity:{&lt;NN.*>&lt;NN.*>+}')
# print(Tree2Chunks(cp.parse(LTsPythonTags)))
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 4
# 
# In `sPython` find chunk phrases which contain noun phrases `<DT>?<JJ><NN>` or `<VB.*><NN.*>`. For example, your grammer should extract the following chunks:
# 
#     [('sole responsibility', 'NP'), ('permanent vacation', 'NP'), ...]
#     
# <b>Hint:</b> When combining phrase definitions, remember to separate these with a newline character.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# cp = nltk.RegexpParser('NP: {&lt;DT>?&lt;JJ>&lt;NN>} \nVP: {&lt;VB.*>&lt;NN.*>}')
# print(Tree2Chunks(cp.parse(LTsPythonTags)))
#     </pre>
#     </details> 
# </font>
# <hr>

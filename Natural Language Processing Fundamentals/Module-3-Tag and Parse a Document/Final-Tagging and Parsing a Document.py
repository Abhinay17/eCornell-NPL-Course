#!/usr/bin/env python
# coding: utf-8

# # **Part Three of the Course Project**
# 
# In this part of the course project, you will apply the skills you have practiced in this module to complete functions in a preprocessing pipeline for tagging and parsing text. You will use each function individually to investigate a mix of large and small documents from several quotes about learning, as well as from a corpus of inaugural speeches given by American Presidents.  
# 
# The preprocessing pipеline for tagging and parsing (i.e. tokenizing) a document you will create follows this workflow: 
#  
# $$\newcommand{\t}{\texttt} 
# \newcommand{\u}[2]{\underset{\texttt{#1}}{\text{#2}}}
# \newcommand{\a}[2]{\underset{\text{#1}\atop\text{#2}}{\longrightarrow}}
# \t{sDoc}\a{parse}{sentences}
# \t{LsSents}\a{parse}{words}
# \t{LLsWords}\a{Penn POS}{tagging}
# \t{LLTsWordPOST}\a{specific}{task}
# \begin{cases} 
#     \t{chunk phrases} \\
#     \t{WordNet POS tags}\rightarrow \t{WordNet word lemmas} \\
#     \t{dependency tree (constituency tree can also be created)} \\
# \end{cases}$$
#  
# where 
#  
# 1. `sDoc` is a string document with at least one sentence
#     1. Ex: a string with 2 sentences, `'I do. We go.'`
# 1. `LsSents` is a list of string sentences
#     1. Ex: `['I do.', 'We go.']`
# 1. `LLsWords` is a list of lists of words of sentences
#     1. Ex: `[['I','do','.'],['We','go','.']]`
# 1. `LLTsWordPOST` is a list of lists of tuples of word & Penn POS tag pairs
#     1. Ex: `[[('I','PRP'),('do','VBP'),('.','.')],  [('We','PRP'),('go', 'VBP'),('.', '.')]]`
# 1. Wordnet lemmatizer uses WordNet POS tags: `'a'`:adjective, `'n'`:noun (and is the default), `'r'`:adverb, `'v'`:verb
#     1. Ex: `[[('I','n'),('do','v'),('.','n')],  [('We','n'),('go', 'v'),('.', 'n')]]`
# 
#     
# Note that the pipeline does not include sentence tokenization because the `spacy` library does sentence tokenization implicitly, so we can feed a whole document to it for processing.
# 
# **Once you have completed this pipeline, you will be equipped to extract phrases, lemmas, and dependencies based on POS tags from the Penn and WordNet tagsets using the `nltk` and `spacy` libraries.**
# <hr style="border-top: 2px solid #606366; background: transparent;">

# # Setup
# 
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries and data sets you will need to complete this part of the course project. 

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS; IS.ast_node_interactivity = "all"
import nltk, pandas as pd, spacy, numpy.testing as npt
from nltk.corpus import inaugural
from typing import List, Tuple, Set
import unittest
from colorunittest import run_unittest
_ = nltk.download(['omw-1.4','wordnet', 'inaugural', 'averaged_perceptron_tagger', 'punkt', 'tagsets'], quiet=True);
pd.set_option('max_rows', 10, 'max_columns', 100, 'max_colwidth', 100, 'precision', 2)
eq, aeq = npt.assert_equal, npt.assert_almost_equal

# Dictionary with epic quotes about education and learning:
DsEdu = {'Albert Einstein':     "Education is what remains after one has forgotten what one has learned in school.",
         'Albert Einstein(2)':  "It’s not that I’m so smart, it’s just that I stay with problems longer.",
         'B. B. King':          "The beautiful thing about learning is that no one can take it away from you.",
         'Benjamin Franklin':   "An investment in knowledge pays the best interest.",
         'Sydney J. Harris':    "The whole purpose of education is to turn mirrors into windows.",
         'Nelson Mandela':      "Education is the most powerful weapon which you can use to change the world.",
         'Dorothy Parker':      "The cure for boredom is curiosity. There is no cure for curiosity.",
         'Mahatma Gandhi':      "Live as if you were to die tomorrow. Learn as if you were to live forever."}

LLsPIWords = inaugural.sents('1841-Harrison.txt') # list of lists of words from Presidential Inaugurations (PI)
LsPISents = [' '.join(words) for words in inaugural.sents('1841-Harrison.txt')]  # list of sentences from PI 
sPIDoc = inaugural.raw('1841-Harrison.txt') 

print(f'LLsPIWords =', str(LLsPIWords[:1])[:100])   # list of lists of sentence word strings
print(f'LsPISents =', str(LsPISents[:1])[:100])     # list of sentence strings
print(f'sPIDoc =', sPIDoc[:100])                    # string document


# Since we will use the [Penn Treebank POS tag set](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html), let's review the full list of tags here.

# In[2]:


# nltk.help.upenn_tagset()  # prints all tags with descriptions
pd.set_option('max_rows', 100, 'display.max_colwidth', 0)
DTagSet = nltk.data.load('help/tagsets/upenn_tagset.pickle')  # dictionary of POS tags
pd.DataFrame(DTagSet, index=['Definition', 'Examples']).T.sort_index().reset_index().rename(columns={'index':'Tag'})


# ## Create the Pipeline
# 
#     
# The series of functions that you'll complete to create the pipeline are below. You'll work through the pipeline step by step. Each function accomplishes a portion of the pipeline. Use the title of each function or group of functions to determine which part of the pipeline you're accomplishing as you complete each function.
# 
# ### Function 1: Parse `sDoc` into Sentences

# In[3]:


from nltk.tokenize import sent_tokenize
def Doc2Sents(sDoc='I like NLP. NLP is fun.') -> List[str]:
    '''Use nltk.sent_tokenize() to parse sDoc into a list of string sentences'''
    return sent_tokenize(sDoc)
    raise NotImplementedError()


# In[4]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestDoc2Sents(unittest.TestCase):
    def test_00(self): eq(Doc2Sents(), ['I like NLP.', 'NLP is fun.'])
    def test_01(self): eq(Doc2Sents(DsEdu['Benjamin Franklin']), ['An investment in knowledge pays the best interest.'])
    def test_02(self): eq(Doc2Sents(DsEdu['Mahatma Gandhi']), ['Live as if you were to die tomorrow.', 'Learn as if you were to live forever.'])


# ### Function 2: Parse Sentence into a List of Words

# In[5]:


from nltk.tokenize import word_tokenize
def Sent2Words(sSent='I like NLP') -> List[str]:
    '''Use nltk.word_tokenize() to parse a single sentence into a list of words'''
    return word_tokenize(sSent)
    raise NotImplementedError()


# In[6]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestSent2Words(unittest.TestCase):
    def test_00(self):  eq(Sent2Words(), ['I', 'like', 'NLP'])
    def test_01(self): eq(Sent2Words(DsEdu['Benjamin Franklin']),         ['An','investment','in','knowledge','pays','the','best','interest', '.'])


# ### Function 3: POS Tag Each Sentence with UPenn Tags

# In[7]:


from nltk import pos_tag
def POST(LsWords=['I','like','NLP']) -> List[Tuple[str]]:
    ''' Use nltk.pos_tag() with default parameters to tag words in a sentence.
    Return: list of tuples in the form (word string, Penn POS tag string)'''
    return pos_tag(LsWords)
    raise NotImplementedError()


# In[8]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestPOST(unittest.TestCase):
    def test_00(self): eq(POST(), [('I', 'PRP'), ('like', 'VBP'), ('NLP', 'NNP')])
    def test_01(self): eq(' '.join([tag for word, tag         in POST(Sent2Words(DsEdu['Mahatma Gandhi']))]),             'NNP IN IN PRP VBD TO VB NN . NNP IN IN PRP VBD TO VB RB .')


# ### Function 4: Create Wordnet POS Tags from UPenn Tags

# In[9]:


def WN_POST(LTsPOST=[('I', 'PRP'), ('like', 'VBP'), ('NLP', 'NNP')]) -> List[Tuple[str]]:
    '''Replace Penn POS tags with WordNet POS tags. 
    Rules: tags 'a','v','r' replace any Penn tags starting with the same letters. 
    Tag 'n' replaces all other tags.
    LTsPOST: list of tuples in the form (word string, Penn POS tag string)
    Return: list of tuples in the form (word string, WordnNet POS tag string)     '''
    result = []
    for word, penn_tag in LTsPOST:
        if penn_tag.startswith('JJ'):
            result.append((word, 'a'))
        elif penn_tag.startswith('VB'):
            result.append((word, 'v'))
        elif penn_tag.startswith('RB'):
            result.append((word, 'r'))
        else:
            result.append((word, 'n'))
    return result
    raise NotImplementedError()


# In[10]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestWN_POST(unittest.TestCase):
    def test_00(self): eq(WN_POST(), [('I', 'n'), ('like', 'v'), ('NLP', 'n')])
    def test_01(self): eq(''.join([tag for word, tag in WN_POST(POST(Sent2Words(DsEdu['Albert Einstein'])))]), 'nvnvnnvvnnvvnnn')
    def test_02(self): eq(''.join([tag for word, tag in WN_POST(POST(Sent2Words(DsEdu['Dorothy Parker'])))]), 'nnnnvnnnvnnnnn')


# ## Specific Pipeline Tasks
# 
# Now that you've completed many of the precursor steps in your pipeline, you're ready to use the preprocessed text to perform specific tasks. 
# 
# ### Function 5: Lemmatize Wordnet-tagged Words

# In[11]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
def Lemmas(LsWN_POST=[('I', 'n'), ('liked', 'v'), ('NLP', 'n')]) -> List[str]:
    ''' Use nltk's WordNet lemmatizer with default parameters to convert 
            a list of WordNet-tagged words to a list of corresponding lemmas. '''
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word, pos_tag in LsWN_POST:
        if pos_tag == 'n':
            lemma = lemmatizer.lemmatize(word, wordnet.NOUN)
        elif pos_tag == 'v':
            lemma = lemmatizer.lemmatize(word, wordnet.VERB)
        elif pos_tag == 'a':
            lemma = lemmatizer.lemmatize(word, wordnet.ADJ)
        elif pos_tag == 'r':
            lemma = lemmatizer.lemmatize(word, wordnet.ADV)
        else:
            lemma = word
        lemmas.append(lemma)
    return lemmas
    raise NotImplementedError()


# In[12]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestLemmas(unittest.TestCase):
    def test_00(self): eq(Lemmas(),['I', 'like', 'NLP'])
    def test_01(self): eq(Lemmas(WN_POST(POST(Sent2Words(DsEdu['Benjamin Franklin'])))),         ['An', 'investment', 'in', 'knowledge', 'pay', 'the', 'best', 'interest', '.'])
    def test_02(self): eq(' '.join(Lemmas(WN_POST(POST(Sent2Words(DsEdu['Albert Einstein']))))),         'Education be what remain after one have forget what one have learn in school .')


# ### **Chunking**
# 
# Next you will apply regex parsers to extract chunk phrases from POS tagged sentences. The following function,  `ChunkNTag()`, is already defined for you. It retrieves labeled chunks from the `nltk` tree structure. Use it to define and explore your own chunk definitions with POS tags. See examples below the function.

# In[13]:


def ChunkNTag(LTsPOST=[('I','PRP'),('like','VB'),('NLP','NN')], sGrammar="VP: {<V.*>+}") -> List[Tuple[str]]:
    '''Given a POS tagged sentence, extracts chunk-tagged phrased based on grammar specification.
    Builds a shallow NLTK tree structure and retrieves chunks under the root of this tree, 
        which is the full sentence with a tag 'S'
    Input:
        LTsPOST: Penn POS-tagged words as a list of tuples of strings in the form (word, tag)
        sGrammar: chunk grammar as a regex pattern string.
    Return: list of tuples of strings of chunk phrase & tag pair '''
    ChunkTree = nltk.RegexpParser(sGrammar).parse(LTsPOST)  # chunk parser returns an NLTK's Tree structure
    # In double-loop below, we convert each subtree to a list of tuples with chunk label and its phrase
    return [(' '.join(leaf[0] for leaf in tree.leaves()), tree.label()) for tree in ChunkTree.subtrees() if tree.label()!='S']

print('Verb phrases in "I like NLP":\t', ChunkNTag())
print('Original quote:\t',DsEdu['Albert Einstein'])
print('Tagged words:\t',  POST(Sent2Words(DsEdu['Albert Einstein'])))
print('Verb phrases:\t',  ChunkNTag(POST(Sent2Words(DsEdu['Albert Einstein'])), sGrammar="VP:{<VBZ><VBN>}"))
print('Wh-word phrases\t',ChunkNTag(POST(Sent2Words(DsEdu['Albert Einstein'])), sGrammar="WP:{<WP><CD>?<V..>*}"))
print('Noun phrases:\t',  ChunkNTag(POST(Sent2Words(DsEdu['Albert Einstein'])), sGrammar="NP: {<NN>}"))
print('In-NN phrases:\t', ChunkNTag(POST(Sent2Words(DsEdu['Albert Einstein'])), sGrammar="NP0: {<IN><NN.*>}"))
print('CD+VP phrases:\t', ChunkNTag(POST(Sent2Words(DsEdu['Albert Einstein'])), sGrammar="CD VP:{<CD><VB.>}"))


# ### Function 6: Chunking

# In[14]:


def Chunk(LTsPOST=[('I','PRP'),('like','VB'),('NLP','NN')], sGrammar="VP: {<V.*>+}", sExtractTag='VP') -> List[str]:
    '''Given a POS tagged sentence, use ChunkNTag() to extract a list 
       of chunk phrases only, filtered by sExtractTag.
    Input:
        LTsPOST, sGrammar: same arguments as in ChunkNTag()
        sExtractTag: chunk tag string identifying chunk phrases to keep in the returned list.
    Return: list of string chunk phrases corresponding to the specified sExtractTag. '''
    chunked_phrases = ChunkNTag(LTsPOST, sGrammar)
    filtered_phrases = [phrase for phrase, tag in chunked_phrases if tag == sExtractTag]
    return filtered_phrases
    raise NotImplementedError()


# In[15]:


# RUN CELL TO TEST YOUR CODE
LTsPOST = POST(Sent2Words(DsEdu['Albert Einstein']))
@run_unittest
class TestChunk(unittest.TestCase):
    def test_00(self): eq(Chunk(LTsPOST, sGrammar="VP:{<V.*>+}", sExtractTag='VP'), ['is','remains','has forgotten','has learned'])
    def test_01(self): eq(Chunk(LTsPOST, sGrammar="VP:{<V.*>+}", sExtractTag='NP'), [])
    def test_02(self): eq(Chunk(LTsPOST, sGrammar="NP:{<NN.*>+}", sExtractTag='NP'), ['Education','school'])
    def test_03(self): eq(Chunk(LTsPOST, sGrammar="NP:{<NN.*>+}", sExtractTag='VP'), [])
    def test_04(self): eq(Chunk(LTsPOST, sGrammar="WP:{<WP><CD>?<V..>*}", sExtractTag='WP'), ['what remains','what one has learned'])
    def test_05(self): eq(Chunk(LTsPOST, sGrammar="CD VP:{<CD><VB.>}", sExtractTag='CD VP'), ['one has','one has'])
    def test_06(self): eq(Chunk(LTsPOST, sGrammar="VP:{<V.*>+}\nNP:{<NN.*>+}", sExtractTag='NP'), ['Education', 'school'])
    def test_07(self): eq(Chunk(LTsPOST, sGrammar="VP:{<V.*>+}\nNP:{<NN.*>+}", sExtractTag='VP'), ['is', 'remains', 'has forgotten', 'has learned'])


# ### Function 7: Chunking with Regex

# In[16]:


def GetNP1(LTsPOST=[('I','PRP'),('like','VB'),('NLP','NN')]) -> List[str]:
    ''' Use Chunk() to find chunk noun phrases (NP1) with
        at most 1 determiner followed by at most 1 adjective 
        followed by at least one noun in any form.
    Input:
        LTsPOST: Penn POS-tagged words as a list of tuples of strings in the form (word, tag)
    Returns a list of chunk noun phrases from the given sentence  '''
    sGrammar = "NP1: {<DT>?<JJ>?<NN.*>+}"
    np1_phrases = Chunk(LTsPOST, sGrammar, sExtractTag="NP1")
    return np1_phrases
    raise NotImplementedError()


# In[17]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestGetNP1(unittest.TestCase):
    def test_00(self): eq(GetNP1(POST(Sent2Words(DsEdu['Albert Einstein']))), ['Education', 'school'])
    def test_01(self): eq(GetNP1(POST(Sent2Words(DsEdu['B. B. King']))), ['The beautiful thing', 'learning', 'no one'])
    def test_02(self): eq(GetNP1(POST(Sent2Words(DsEdu['Benjamin Franklin']))), ['An investment', 'knowledge', 'interest'])
    def test_03(self): eq(GetNP1(POST(Sent2Words(DsEdu['Sydney J. Harris']))), ['The whole purpose', 'education', 'mirrors', 'windows'])
    def test_04(self): eq(GetNP1(POST(Sent2Words(DsEdu['Albert Einstein(2)']))), ['m', 'problems'])


# ### Function 8: Chunking with Regex

# In[18]:


def GetNP2(LTsPOST=[('I','PRP'),('like','VB'),('NLP','NN')]) -> List[str]:
    ''' Use Chunk() to find chunk noun phrases (NP2) with
        at exactly 1 determiner followed by at most 1 adjective followed by at least one noun in any form.
    Input:
        LTsPOST: Penn POS-tagged words as a list of tuples of strings in the form (word, tag)
    Returns a list of chunk noun phrases from the given sentence that have  '''
    sGrammar = "NP2: {<DT><JJ>?<NN.*>+}"
    np2_phrases = Chunk(LTsPOST, sGrammar, sExtractTag="NP2")
    return np2_phrases
    raise NotImplementedError()


# In[19]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestGetNP2(unittest.TestCase):
    def test_00(self): eq(GetNP2(POST(Sent2Words(DsEdu['Albert Einstein']))), [])
    def test_01(self): eq(GetNP2(POST(Sent2Words(DsEdu['B. B. King']))), ['The beautiful thing', 'no one'])
    def test_02(self): eq(GetNP2(POST(Sent2Words(DsEdu['Benjamin Franklin']))), ['An investment'])
    def test_03(self): eq(GetNP2(POST(Sent2Words(DsEdu['Sydney J. Harris']))), ['The whole purpose'])
    def test_04(self): eq(GetNP2(POST(Sent2Words(DsEdu['Albert Einstein(2)']))), [])


# ### Function 9: Chunking with Regex

# In[20]:


def GetNP3(LTsPOST=[('I','PRP'),('like','VB'),('NLP','NN')]) -> List[str]:
    ''' Use Chunk() to find chunk noun phrases (NP3) with
        exactly 1 noun (in any form) exactly 1 verb (in any form, with POS tag starting with VB).
    Input:
        LTsPOST: Penn POS-tagged words as a list of tuples of strings in the form (word, tag)
    Returns a list of chunk noun phrases from the given sentence that have  '''
    sGrammar = "NP3: {<NN.*><VB.*>}"
    np3_phrases = Chunk(LTsPOST, sGrammar, sExtractTag="NP3")
    return np3_phrases
    raise NotImplementedError()


# In[21]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestGetNP3(unittest.TestCase):
    def test_00(self): eq(GetNP3(POST(Sent2Words(DsEdu['Albert Einstein']))), ['Education is'])
    def test_01(self): eq(GetNP3(POST(Sent2Words(DsEdu['B. B. King']))), ['learning is'])
    def test_02(self): eq(GetNP3(POST(Sent2Words(DsEdu['Benjamin Franklin']))), ['knowledge pays'])
    def test_03(self): eq(GetNP3(POST(Sent2Words(DsEdu['Sydney J. Harris']))), ['education is'])
    def test_04(self): eq(GetNP3(POST(Sent2Words(DsEdu['Albert Einstein(2)']))), [])


# ### Function 10: Chunk Frequency

# In[22]:


from collections import Counter
def FreqNP1Phrase(LLTsPOST=[[('I','PRP'),('like','VBP'),('math','NN')]], 
                  nMinFreq=1, nMinChar=1) -> List[Tuple[str,int]]:
    '''Use collections.Counter().most_common() and GetNP1() to find 
        the most frequent NP1 chunks in LLTsPOST and of restricted length.
    Input:
        LLTsPOST: list of lists of tuples of strings with (word, Penn POS tag) format
        nMinFreq: min frequency for the chunk phrases
        nMinChar: the minimum allowed number of characters in the chunk phrase
    Return: list of tuples of (phrase string, count integer)  '''
    np1_chunks = []
    for sentence in LLTsPOST:
        np1_chunks.extend(GetNP1(sentence))
    counter = Counter(np1_chunks)
    filtered_chunks = [(phrase, count) for phrase, count in counter.items() 
                        if count >= nMinFreq and len(phrase) >= nMinChar]
    return sorted(filtered_chunks, key=lambda x: x[1], reverse=True)
    raise NotImplementedError()


# In[23]:


# RUN CELL TO TEST YOUR CODE
LTsEdu = [POST(Sent2Words(s)) for s in Doc2Sents(' '.join(DsEdu.values()).lower())]
@run_unittest
class TestFreqNP1Phrase(unittest.TestCase):
    def test_00(self): eq(FreqNP1Phrase(LTsEdu[0:1]), [('education', 1), ('school', 1)])
    def test_01(self): eq(FreqNP1Phrase(LTsEdu[0:2]), [('education', 1), ('school', 1), ('m', 1), ('problems', 1)])
    def test_02(self): eq(FreqNP1Phrase(LTsEdu[0:5], 2), [('education', 2)])
    def test_03(self): eq(FreqNP1Phrase(LTsEdu, 2), [('education', 3), ('curiosity', 2)])
    def test_04(self):
        LTsPI_POST = [POST(Sent2Words(s)) for s in LsPISents] # let's test on presidential inaugural speech
        LTsnPIOut = [('the Constitution',31), ('the people', 24), ('the Government', 12), ('the character', 10), ('the Executive', 10)]
        eq(FreqNP1Phrase(LTsPI_POST, nMinFreq=10, nMinChar=10), LTsnPIOut)


# ### **Dependency Parsing**
#  
# Finally, you will extract the dependency tree from a sentence. In this case, SpaCy does most of the parsing and tagging work for you, you just have to extract specific components from its `nlp` object.
# 
# ### Function 11: Spacy Tagging

# In[24]:


nlp = spacy.load("en_core_web_sm")   # SpaCy's text-processing pipeline object for English
def DTRoot(sSent='I like NLP') -> List[str]:
    '''Find all roots of SpaCy's dependency trees from sentence.
    Note: for complex sentences SpaCy returns multiple trees with correpsonding roots.
    Input: sSent: sentence string
    Return: list of roots (strings) '''
    doc = nlp(sSent)
    roots = [token.text for token in doc if token.dep_ == 'ROOT']
    return roots
    raise NotImplementedError()


# In[25]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestDTRoot(unittest.TestCase):
    def test_00(self):eq(DTRoot(), ['like'])
    def test_01(self): eq([DTRoot(s) for s in DsEdu.values()],         [['is'],['’s'],['is'],['pays'],['is'],['is'],['is', 'is'],['Live', 'Learn']])


# ### Function 12: Root Frequency

# In[26]:


def FreqDTRoot(sDoc='I like NLP', nMinFreq=1, nMinChar=2) -> List[Tuple[str,int]]:
    '''Use collections.Counter.most_common() and DTRoot() to find 
        most frequent roots from a document string with multiple sentences.
        Filter out roots shorter than nMinChar.
    Input: 
        sDoc: a string with multiple sentences, which need to be split with Doc2Sents()
        nMinFreq: min frequency for the root word
        nMinChar: the minimum allowed number of characters in the root word
    Return: list of tuples of (root string, count integer) format   '''
    doc = nlp(sDoc)
    roots = []
    for sent in doc.sents:
        roots.extend(DTRoot(sent.text))
    counter = Counter(roots)
    filtered_roots = [(root, count) for root, count in counter.items()
                      if count >= nMinFreq and len(root) >= nMinChar]
    return sorted(filtered_roots, key=lambda x: x[1], reverse=True)
    raise NotImplementedError()


# In[27]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestFreqDTRoot(unittest.TestCase):
    def test_00(self): eq(FreqDTRoot(' '.join(DsEdu.values()), 2), [('is', 6)])
    def test_01(self): eq(FreqDTRoot(sPIDoc, 3, 6),         [('become', 6), ('appear', 4), ('observed', 3), ('appears', 3)])


# In[ ]:





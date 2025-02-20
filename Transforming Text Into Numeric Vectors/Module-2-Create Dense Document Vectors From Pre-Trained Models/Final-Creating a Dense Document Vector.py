#!/usr/bin/env python
# coding: utf-8

# # Part Two of the Course Project
# 
# <span style="color:black">In this part of the course project, you'll complete a set of functions that retrieve word vectors from a Word2Vec model, process the model's vocabulary to work better with similarity analyses, and then use these functions to analyze similarity of pairs and groups of words. As you use these functions, you will work with the <b>glove-wiki-gigaword-50</b> pre-trained Word2Vec model that you've worked with in this module. 
#     
# <p style="color:black">Begin by loading the required libraries and printing the versions of NLTK, Gensim, and NumPy using their <code>__version__</code> attribute.</p>
# 
# <p style="color:black"><b>Note:</b> Since word-embedding models are a rapidly changing area of NLP, changes in library versions may break older code. Pay attention to library versions and, as always, carefully read error messages. We will note where the functionality diverges from that demonstrated in the videos and provide alternative methods you can use to complete the task. 
#     
# <hr style="border-top: 2px solid #606366; background: transparent;">

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete this part of the course project. 

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS; IS.ast_node_interactivity = "all"
import numpy as np, pandas as pd, numpy.testing as npt, nltk, gensim
from gensim.models import KeyedVectors
import unittest
from colorunittest import run_unittest
eq, aeq, areq = npt.assert_equal, npt.assert_almost_equal, np.testing.assert_array_equal

# Expected Codio versions: NLTK 3.6.2, gensim 4.0.1, np 1.19.5
print(f'Versions. nltk:{nltk.__version__}, gensim:{gensim.__version__}, np:{np.__version__}')  


# Next, Word2Vec model (in compressed gz format) is loaded from the local Jupyter folder.
# 
# ### **Note:** This model may take between 30 and 60 seconds to load.

# In[2]:


# Dictionary-like object. key=word (string), value=trained embedding coefficients (array of numbers)
# https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz
get_ipython().run_line_magic('time', "wv = KeyedVectors.load_word2vec_format('glove-wiki-gigaword-50.gz')")
wv            # prints the type of the object and its memory location


# # **Function 1: Retrieve Word Vectors**
# 
# Complete this function so that it extracts the word vector for a given word from the `wv` Word2Vec model.
# 
# ### Note: This function may take 30 seconds or longer to run when complete.

# In[3]:


def GetWV(wv, sWord='nlp') -> np.array:
    ''' Returns a word vector for sWord (in lower case), if it is found, 
        and a zero vector (of length n) otherwise, where n is the length of vectors in wv.
    wv: Gensim's word2vec model object'''
    sWord = sWord.lower()
    if sWord in wv:
        return wv[sWord]
    else:
        return np.zeros(wv.vector_size)


# In[4]:


# TEST & AUTOGRADE CELL
@run_unittest
class test_GetWV(unittest.TestCase):
    def test_00(self): eq(GetWV(wv, 'Los Angeles').sum(), 0.0)
    def test_01(self): aeq(GetWV(wv, 'Cornell').sum(), -1.7181, 4)
    def test_02(self): aeq(GetWV(wv, 'nlp').sum(), 5.4109, 4)


# # Function 2: Extract Qualifying Words
# 
# Complete the `GetSupWords` function so it searches through `wv` lexicon and extracts "qualifying" words. Qualifying words are those that are lowercased substrings of a given word `sWord`.
# 
# Once you have completed this function, spend some time exploring the word tokens in the dictionary. You will find that many of them are not words at all but numbers, phone numbers, punctuation symbols, and various word parts. Also compound words, such as "english-language," "york-new," "new-york" are stored. It's an important observation because in order to obtain a vector for "new york," we would need to first identify "new" and "york" as part of a single word and then add a hyphen to bring it to a word form, for which the vector can be found. If we simply parse our text on spaces, we would end up with two vectors: one for "new" and one for "york," which are vaguely related to the state of New York and New York City. So, whenever working with a Word2Vec model, spend some time to understand the distribution of words and their forms in the model's vocabulary.

# In[5]:


def GetSupWords(wv, sWord='nlp') -> [str]:
    '''Return all wv vocabulary words, for which sWord (in lower case) 
        is a subword, i.e. a substring. If none is found, return an empty list.
    wv: Gensim's word2vec model '''
    sWord = sWord.lower()
    return [word for word in wv.index_to_key if sWord in word]


# In[6]:


# TEST & AUTOGRADE CELL
@run_unittest
class test_GetSupWords(unittest.TestCase):
    def test_00(self): eq(GetSupWords(wv, 'CatsFromSpace'), [])
    def test_01(self): eq(GetSupWords(wv, 'NLP'), ['nlp'])
    def test_02(self): eq(GetSupWords(wv, 'Cornell'), ['cornell', 'cornella', 'cornellà'])
    def test_03(self): eq(GetSupWords(wv, 'York')[:6], ['york', 'yorkshire', 'yorker', 'yorkers', 'yorke', 'yorktown'])
    def test_04(self): eq(GetSupWords(wv, 'language')[:4], ['language', 'languages', 'english-language', 'spanish-language'])
    def test_05(self): eq(GetSupWords(wv, '123')[:7], ['123', '1230', '1234', '123,000', '1235', '1236', '1237'])


# # **Function 3: Find Nearest Neighbors**
# 
# Here, you will complete the function `NN` to modify the `wv.most_similar()` method, which retrieves `topn` most similar words for the given "positive" word so that it retrieves only most similar words that are within `nThreshold`, the similarity to `sWord`.
# 
# This modification is practical in situations where `sWord` (some rare word) has too few neighbors so the most similar function returns irrelevant words. For example, applying a threshold on `wv.most_similar("gulag", topn=10)` retrieves words that rapidly drop in similarity from 0.77 to 0.58. Thus, applying a threshold on similarity so that we don't retrieve irrelevant words would be more suitable.
# 
# Hint: Set `topn` within `vw.most_similar()` as the length of the whole vocabulary then apply a threshold to the similarity value.

# In[7]:


def NN(wv, sWord='pet', nThreshold=0.75) -> [(str, float)]:
    '''For sWord (in lower-case), return a list of most similar words 
    and corresponding similarity score. Only similarities above nThreshold are returned.
    If none is found, return an empty list.
    Inputs: 
        wv: Gensim's word2vec model object
        sWord: string word for which most semantically similar words are retrieved 
        nThreshold: fraction of similar words to retrieve for sWord
    Returns: returns a list of tuples (word, similarity score) from the .most_similar() method '''
    sWord = sWord.lower()
    if sWord in wv:
        vocab_size = len(wv.index_to_key)
        similar_words = wv.most_similar(sWord, topn=vocab_size)
        return [(word, score) for word, score in similar_words if score > nThreshold]
    return [] 
print(NN(wv, 'language', 0.80))


# In[8]:


# TEST & AUTOGRADE CELL

@run_unittest
class test_NN(unittest.TestCase):
    def test_00(self): eq(type(NN(wv, 'not found')), list)
    def test_01(self): eq( NN(wv, 'x men'), [])
    def test_02(self): areq(NN(wv, 'Cornell', 0.85), [('yale', 0.8834298253059387), 
                          ('harvard', 0.8587191104888916), ('princeton', 0.8516749739646912)])
    def test_03(self): areq(NN(wv, 'language', 0.85), [('languages', 0.8814865946769714)])
    def test_04(self): areq(NN(wv, 'language', 0.80), [('languages', 0.8814865946769714), 
                           ('word', 0.8100197315216064), ('spoken', 0.8074647784233093)])
    def test_05(self): aeq(sum([s for _,s in NN(wv, 'Cornell', 0.75)[:5]]), 4.228408098220825, 4)
    def test_06(self): eq(','.join([w for w,_ in NN(wv, 'language', 0.75)[:5]]), 'languages,word,spoken,vocabulary,translation')
    def test_07(self): aeq(sum([s for _,s in NN(wv, 'language', 0.75)[:5]]), 4.077211260795593, 4)
    def test_08(self): eq(','.join([w for w,_ in NN(wv, 'english-language', 0.75)[:5]]),         'german-language,french-language,spanish-language,russian-language,arabic-language')
    def test_09(self): aeq(sum([sim for _,sim in NN(wv, 'english-language', 0.75)[:5]]), 4.016840815544128, 4)


# # **Function 4: Find a Pair of Neighbors**
# 
# Complete this function, `NN2`, so that it identifies the pair of words that are semantically the closest in a given list, `LsWords`. Pay attention to ordering. Convert all words in `LsWords` to lowercase.
# 
# For example, the following call:
# 
#     NN2(wv, 'Cat Ant Rat Owl Dog Cow Pig Hen Ape Man Elk Bee Eel Fox Bat Emu Gnu Koi'.split())
#     
# should return:
# 
#     (0.9218005, 'cat', 'dog')

# In[9]:


def NN2(wv, LsWords=['cat','dog','NLP']) -> (float, str, str):
    ''' Given a list of words in LsWords, identify a pair of semantically-closest (lower-cased) words.
        Use Gensim's similarity() method (i.e. cosine similarity) to measure closeness.
        If the count of words (for which vectors are available)<2, return None
    wv: Gensim's word2vec model
    Return as a tuple containing a similarity score, and two (lower-cased) strings, 
        each containing one of the pair of closest words in alphabetical order. '''
    
    ##### Pseudocode Hints #####
    
    # Step 1: Keep only lower-cased words which are in the word2vec vocabulary
    
    # Step 2: Make sure the resulting list contains at least two words, otherwise return None
    
    # Step 3: If step 2 is passed, compare the similarity score of every possible pair of words
    #         in the list and return the two words with the highest similarity score as described
    #         in the instructions above the function
    
    
    valid_words = [word.lower() for word in LsWords if word.lower() in wv]
    if len(valid_words) < 2:
        return None
    max_similarity = -1
    closest_pair = ("", "")
    for i in range(len(valid_words)):
        for j in range(i + 1, len(valid_words)):
            word1, word2 = valid_words[i], valid_words[j]
            similarity = wv.similarity(word1, word2)
            #print(f"Comparing: {word1} - {word2}, Similarity: {similarity}")
            if similarity > max_similarity:
                max_similarity = similarity
                closest_pair = tuple(sorted([word1, word2]))
    print(f"Max Similarity: {max_similarity}, Closest Pair: {closest_pair}")
    return (max_similarity, closest_pair[0], closest_pair[1])


# In[10]:


# TEST & AUTOGRADE CELL
LsWords1 = 'Cat Ant Rat Owl Dog Cow Pig Hen Ape Man Elk Bee Eel Fox Bat Emu Gnu Koi'.split()
LsWords2 = [w for w,_ in NN(wv, 'Pet', 0.7)]
LsWords3 = [w for w,_ in NN(wv, 'google', 0.7)]
LsWords4 = [w for w,_ in NN(wv, 'english-language', 0.6)]
LsWords5 = [w for w,_ in NN(wv, 'university', 0.7)]

@run_unittest
class test_NN2(unittest.TestCase):
    def test_00(self): eq(type(NN2(wv, ['cat','dog'])), tuple)   # ensure that tuple is returned
    def test_01(self): eq(NN2(wv, []), None)
    def test_02(self): eq(NN2(wv, ['cat']), None)
    def test_03(self): areq(NN2(wv, ['cat','rat']), (0.7891964, 'cat', 'rat'))
    def test_04(self): areq(NN2(wv, LsWords1), (0.9218005, 'cat', 'dog'))
    def test_05(self): areq(NN2(wv, ['tom','and','jerry']), (0.74370354, 'jerry', 'tom'))
    def test_06(self): eq(NN2(wv, ['Tom','and_','Jerrry']), None)
    def test_07(self): areq(NN2(wv, LsWords2), (0.9218005, 'cat', 'dog'))
    def test_08(self): areq(NN2(wv, LsWords3), (0.9377265, 'facebook', 'myspace'))
    def test_09(self): areq(NN2(wv, LsWords4), (0.9159014, 'polish-language', 'russian-language'))
    def test_10(self): areq(NN2(wv, LsWords5), (0.95974344, 'harvard', 'yale'))


# # **Function 5: Find Neighbors With Conditions**
# 
# Complete the function `NNExc` so it finds the words that are most similar to the given word and are not on the exception list. This is a helper function that you will use in Function 6. This helper function should return the list of the most similar words for a given word, `sWord`. This list cannot include words in the exception list, `LsExcept`.

# In[11]:


def NNExc(wv, sWord='pet', LsExcept=['cat', 'dog']) -> (str, float):
    ''' 
    Lower-case all input words and use Gensim's most_similar() 
    to find sWord's neighbor, which is not in LsExcept list.
    wv: Gensim's word2vec model  
    Return: a tuple with (neighbor X, similarity score between X and sWord)
    If none is found, return None.
    '''
    sWord = sWord.lower()
    try:
        similar_words = wv.most_similar(sWord, topn=10)  # Get top 10 most similar words
        for word, similarity in similar_words:
            if word not in LsExcept:
                return (word, similarity)
        return None
    except KeyError:
        return None


# In[12]:


# TEST & AUTOGRADE CELL
@run_unittest
class test_NNExc(unittest.TestCase):
    def test_00(self): eq(NNExc(wv, 'pet-', []), None)
    def test_01(self): eq(NNExc(wv, 'Pet', []), ('pets', 0.8009442687034607))
    def test_02(self): eq(NNExc(wv, 'pet', ['pets']), ('cat', 0.7792248725891113))
    def test_03(self): eq(NNExc(wv, 'pet', ['pets','cat']), ('dog', 0.7724707722663879))
    def test_04(self): eq(NNExc(wv, 'pet', ['pets','cat','dog']), ('animal', 0.7471762895584106))


# # **Function 6: Build a Chain of Neighbors**
# 
# Complete the function, `NNChain`, so it builds a sequence of unique words in which subsequent words are semantically the closest to the previous word. The sequence should start with the given word, `sWord`, and end at the specified length, `n`.

# In[13]:


def NNChain(wv, sWord='pet', n=5) -> [(str, float)]:
    ''' For the lower-cased sWord find a chain of n words where each word is the closest
        neighbor of the previous word excluding all words chained so far, including sWord.
        Use NNExc() to find the next neighbor given words in a chain + sWord as the exclusion list.
    Example: 'cat' neighbors with 'dog' with similarity .92; 
             'dog' neighbors with 'dogs' ('cat' was already used), and so on.
    Return a list of chained words with their corresponding similarity scores 
            (between the word and its previous neighbor).
        If none is found, return en empty list. '''
    chain = [(sWord, None)]
    exclusion_list = [sWord]
    current_word = sWord
    for _ in range(n):
        result = NNExc(wv, current_word, exclusion_list)
        if result is None:
            break
        neighbor, similarity = result
        chain.append((neighbor, similarity))
        exclusion_list.append(neighbor)
        current_word = neighbor
    return chain[1:]


# In[14]:


# TEST & AUTOGRADE CELL
@run_unittest
class test_NNChain(unittest.TestCase):
    def test_00(self): eq(NNChain(wv, 'x men', 3), [])
    def test_01(self): eq(NNChain(wv, 'word', 3), [('phrase', 0.9161801934242249), ('phrases', 0.8355081081390381), ('words', 0.8024383187294006)])
    def test_02(self): eq(NNChain(wv, 'cornell', 3), [('yale', 0.8834298253059387), ('harvard', 0.9597433805465698), ('princeton', 0.9076478481292725)])
    def test_03(self): eq(NNChain(wv, 'yosemite', 3), [('yellowstone', 0.7428672909736633), ('elk', 0.7619157433509827), ('beaver', 0.8251944780349731)])
    def test_04(self): eq(NNChain(wv, 'apple', 3), [('blackberry', 0.7543067336082458), ('iphone', 0.7549240589141846), ('ipad', 0.9405524730682373)])
    def test_05(self):
        sOut = ', '.join(list(zip(*NNChain(wv, 'avengers', 10)))[0])
        eq(sOut, 'x-men, wolverine, sabretooth, nightcrawler, psylocke, shadowcat, takiko, baughan, wanley, couvreur')


# ## Conclusion
# 
# In this assignment, you practiced the use of the Word2Vec model in identifying semantically similar words. The Gensim library already gives you tools to find semantically similar words. Here is an example.

# In[15]:


wv.most_similar('doctor', topn=10)


# But you may be interested in extracting a different pattern of collected knowledge from Word2Vec. In the project, you also have built a function to extract a sequence of words semantically related to each other, not to the original query word. In some cases, this may give you a wider range of synonyms or thesaurus. In other cases, if your Word2Vec is built on a specific domain, you may find a more relevant sequence of concepts. For example, in the medical domain, you may not need all synonyms for a doctor but may be interested in treatments, medications, tools, and hospitals relating to the query concept. Here is an example.

# In[16]:


NNChain(wv, 'doctor', 10)


# With these tools in your toolbox, you now have the skills to extract knowledge from Word2Vec and similar knowledge bases, the kinds of concepts that make your work more effective and more fun!

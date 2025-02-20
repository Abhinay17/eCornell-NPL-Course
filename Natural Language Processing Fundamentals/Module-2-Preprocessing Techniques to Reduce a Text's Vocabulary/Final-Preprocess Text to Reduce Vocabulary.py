#!/usr/bin/env python
# coding: utf-8

# # Part Two of the Course Project
# 
# 
# In this part of the course project, you will complete the `Pipe` class so that it can be used to build custom preprocessing pipelines. The `Pipe` class has been partially completed already, but you will need to complete the class attributes, methods, and properties to make this class fully functional. Most of the solutions you will need to write are one-liners, but several may take a few lines.
# 
# The class methods containing preprocessing code are exposed as properties (with `@property` decorator). The properties can be called without parenthesis, which is convenient and visually attractive. Every preprocessing step logs the task name and some basic stats to the dictionary `DStat`, which is stored internally in the instantiated `Pipe` object. So, if needed, one can evaluate the compression of the original document's lexicon at each step of the pipeline.
# <hr style="border-top: 2px solid #606366; background: transparent;">

# # Setup
# 
# To complete this project, you will need to import the `nltk`, `pandas`, and `contractions` libraries.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS; IS.ast_node_interactivity = "all"
import nltk, pandas as pd, numpy.testing as npt, unicodedata, contractions, re
from numpy.testing import assert_equal as eq
import unittest
from colorunittest import run_unittest
_ = nltk.download(['omw-1.4','brown','wordnet','stopwords','averaged_perceptron_tagger'], quiet=True)
from nltk.corpus import brown, stopwords
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ


# ## Background
#     
# This project requires an understanding of [classes](https://docs.python.org/3/tutorial/classes.html) in object-oriented programming. If you are familiar with this concept, you can skip this section.
#     
# In Module 1, you were introduced to the concept that everything in Python is an *object*. Recall that strings are objects with built-in *methods*, e.g., `.join()`, `.split()`, `isalpha()`. For any string, you can call `.split()` to get a list of strings split at whitespace. This is because all string objects are created from the same *class*, i.e., a blueprint for objects.
#     
# Creating an object, often referred to as an *instance*, from a class automatically calls the (`__init__`) method of that class. The initializer accepts `self`, which references the current instance, and other variables to set the class *attributes*, i.e., variables that are shared among all instances. When `self` is passed to other methods/properties **within** the class, they will have access to the variables stored at the class level. For example, if we create two instances, `a=Pipe(...)` and `b=Pipe(...)`, the `self` object for `a` can only access all variables created inside `a` and does not have access to any variables created inside `b`. This encapsulation functionality is very useful for classes. If you feel rusty about class definitions in Python, review your Python prerequisite material or numerous online resources on this topic.
#     
# Object-level variables are accessible via dot notation in Python. A dataframe `df` is an object-level variable accessible via `self.df` inside the object and `Pipe(...).df` outside of the object `Pipe()`. Methods can also be accessed in a similar manner by using `@property` decorator in the class. </span>

# ## Your Tasks 
# 
# * **Task 1**: Initialize attributes  
# Initialize the class attributes (`self.LsWords`, `self.DStat`, `self.LsStep`) in `__init__()`.
#     
# *  **Task 2**: Format Output String 
# <br>Complete the `Out()` method to format the output string.
# 
# *  **Tasks 3 - 10**: String Preprocessing Methods 
# <br> Complete the `Low()`, `NoNum()`, `Words()`, `Stop()`, `Norm()`, `Exp()`, `Stem()`, `Lem()` properties.
#     
# 
# ## Checking Your Work
# 
# Test cases are provided below the project code cell. The `Pipe` class you'll be writing is complex and includes many tasks, so you may want to check whether some methods work before you have completed other methods. You can test select methods in the test cases without completing all of the methods in the class. The final text case will test the full functionality of the `Pipe` class. 

# ## Expected Functionality Examples
# 
# ### Example 1: 
#   
#     >>> LsDoc = "I enjoy learning NLP".split()
#     >>> Pipe(LsDoc, SsStopWords='nltk').Low.Stop.Stem.Out()
#     "enjoy learn nlp"
#  
# In this example, the following properties are applied sequentially to the list of strings in `LsDoc` ["I", "enjoy", "learning", "NLP"]:
# 1. `Low`:  applies lower casing to each word
# 1. `Stop`: removes stop words
# 1. `Stem`: applies PorterStemmer to each word
# 1. `Out()` method returns a preprocessed sentence `"enjoy learn nlp"`
#  
# Notice how the preprocessing steps are stitched with a period (`.Low.Stop.Stem`), where each property returns a reference to the `self` object so that another step can be added.
#     
# ### Example 2:
# 
#     >>> LsDoc = "We'rè fighting CÓVÍD-19 in 2020; ánd we've WON!".split()
#     >>> pp = Pipe(LsDoc, SsStopWords='nltk', SsLex='nltk').Low.Norm.Exp.Stop.Lem.NoNum.Words
#     >>> pp.Out()
#     >>> pp.df
# 
# The following data frame is printed with the step and the corresponding statistics. 
# 
# 
# |.|Step|Words|Vocab|CorrVocab|
# |-|-|-|-|-|
# |0|Initialize|8|8|3|
# |1|Lower-case|8|8|3|
# |2|Normalize accented characters|8|8|5|
# |3|Contraction expansion|10|9|6|
# |4|Remove stopwords|4|4|1|
# |5|Lemmatize|4|4|1|
# |6|Remove numbers|4|4|2|
# |7|Remove non-word characters|4|4|2|

# <hr>
# 
# # `Pipe` Class Code Cell
# 
# The following cell contains the `Pipe` class you'll complete. Right now, each property in the class is folded  to make it easier for you to orient yourself to the class. Click the arrows to the left of the text to unfold the part of the class you want to examine.

# In[2]:


# COMPLETE THIS CELL
class Pipe():
    '''Pipe class exposes several common preprocessing steps as object properties/methods,
    which can be stitched into desirable NLP pipelines using a object's dot notation. 
    For example, 
            Pipe(LsDoc).Low.Stop.Stem.Words.Out() takes LsDoc list of words,
    and passes it through NLP sequence of steps:
            lower-casing -> stop word removal -> stemming -> join words into a document
    In the process, each method accumulates basic statistics from the current list of words'''

    ### TASK 1: Attribute Initialization ###########################################
    def __init__(self, LsWords=[], SsLex=set(), SsStopWords=set()) -> object:
        '''Class constructor. Object-scope variables are initialized here. 
            Then we call AddStats with 'Initialize' argument to save current statistics for LsWords.
        Input:
            LsWords: List[str], string tokens of a document that needs preprocessing
            SsLex:   Set[str] or 'nltk'. 
                A lexicon (set of lower-cased words) to be used for spell checking.
                For 'nltk', we use the set of lower-case words from Brown corpus.
            SsStopWords: Set[str] or 'nltk'. 
                A list of lower-case words to be used as stopwords.
                For 'nltk', we use the set of NLTK English stopwords.
        Returns: reference to self object    '''
        # Ensure correct data structures are passed into the object's initialization method
        assert isinstance(LsWords, list) or LsWords is None, f'LsWords must be a list, not a {type(LsWords)}'
        assert isinstance(SsLex, set) or (SsLex=='nltk'), f'SsLex must be "nltk" or a set of lexicon words, not a {type(SsLex)}'
        assert isinstance(SsStopWords, set) or (SsStopWords=='nltk'), f'SsStopWords must be "nltk" or a set of words, not a {type(SsStopWords)}'

        # df stores preprocessing step name and associated statistics. 
        # We declare a blank object-level dataframe with 4 columns:
        self.df = pd.DataFrame(columns = ['Step', 'Words', 'Vocab', 'CorrVocab'])
        
        # Save each __init__ input value to the object's variable with the same name.  
        # Implement default cases for SsLex & SsStopWords as described in docstring above.

        _ = nltk.download(['brown'], quiet=True)
        Ss6 = {s.lower() for s in nltk.corpus.brown.words()}
        
        # No code changes are necessary for Task 1
        # The print statements below are examples that may be helpful for troubleshooting
        # You may uncomment them and use them, or use additional / different print statements
        # If used, just be sure to re-comment when you are finished troubleshooting 
        # to reduce output
        
        #print("finished lower-casing of nltk corpus brown words")
        #print("first 10 of Ss6 {}".format(Ss6[:10]))
        #print("Ss6 length {}".format(len(Ss6)))
        
        self.LsWords = LsWords
        if SsLex =='nltk':
            self.SsLex = Ss6
            #print("using base nltk lexicon")
        else: 
            self.SsLex = SsLex
            #print("using custom lexicon {}".format(SsLex))
        if SsStopWords =='nltk':
            self.SsStopWords = set(stopwords.words('english'))
            #print("using base nltk stopwords")
            #print("stopwords = {}".format(self.SsStopWords))
        else: 
            self.SsStopWords = SsStopWords
            #print("using custom stopwords")

        self.AddStats('Initialize')     # Saves basic stats for LsWord
        
    ### TASK 2: Output ###########################################
    def Out(self) -> str:
        '''Use string's join method to concatenate words in self.LsWords, 
            separated by a single space. Before returning the string, 
            replace any instance of multi-whitespace with ' '.
            Whitespace characters (space, \t, \r, \n) are represented by \s in regex.
        Returns: a string of single-space-separated cleaned words
        
        For reference, review these pages in the course from Module 1:
        - Preprocess Substrings with Operations
        - Practice Preprocessing Substrings with Operations
        - Overview of Regular Expressions
        - Practice Using Simple Expressions'''
        
        # The return statement in this function is provided as a guide. 
        # You may uncomment and use if it fits your approach, or ignore it if not
        # In other functions, return statements have been provided for you.
        # You will perform operations on self.LsWords per the function specifications
        # and then return the entire self object when complete
        # If you are unfamiliar with this type of coding, you may refer to
        # the AddStats function at the bottom of this cell as a reference
        
        # Once again, some sample print statements have been provided for this function which
        # you may uncomment and use if you find them useful, and you may ignore them if not needed
        # You can implement similar print statements in the other functions as well if desired
        # Remember to comment out all print statements when you have finished troubleshooting
        # in order to minimize unnecessary print output upon submission of your assignment
        
        #print(********)
        #print("running the Out function")
        #print(********)
        #print("self.LsWords starting point = {}".format(self.LsWords))
        
        # YOUR CODE HERE
        return re.sub(r'\s+', ' ', ' '.join(self.LsWords)).strip()
        
        
        
        
        #print(********)
        #print("modified self.LsWords = {}".format(self.LsWords))
        #print(********)
        
        #return self
        
                
        # YOUR CODE HERE
        raise NotImplementedError()
        
    ### TASK 3: Lowercase ###########################################
    @property
    def Low(self) -> object:
        '''Applies lower casing to each word token in self.LsWords and 
            saves results back to self.LsWords.
        Returns: reference to self object for continued chaining of properties 
        
        For reference, review the downloadable tool from Module 1:
        - String Manipulation Methods'''
        
        # YOUR CODE HERE 
        self.LsWords = [word.lower() for word in self.LsWords]
        # YOUR CODE HERE
        # raise NotImplementedError()
        
        return self.AddStats('Lower-case') # finally, we save basic stats after this preprocessing

        
    ### TASK 4: Remove Digits ###########################################
    @property
    def NoNum(self) -> object:
        ''' Use use re.sub() to remove all digits from strings in self.LsWords 
            and save results back to self.LsWords
        In general, the impact of removal of numbers needs to be carefully investigated.
        Returns: reference to self object for continued chaining of properties 
        
        For reference, review these pages in the course from Module 1:
        - Overview of Regular Expressions
        - Practice Using Simple Expressions'''
        
        # YOUR CODE HERE 
        
        
        
        self.LsWords = [re.sub(r'\d+', '', word) for word in self.LsWords]
        
        # YOUR CODE HERE
        # raise NotImplementedError()
        
        return self.AddStats('Remove numbers') # finally, we save basic stats after this preprocessing

        
    ### TASK 5: Keep Only Word Characters ###########################################
    @property
    def Words(self) -> object:
        '''Use re.sub() to keep word characters ('\w': letters, numbers, underscore) and spaces
            only in self.LsWords. Save results back to self.LsWords.
        Note: Removing quotation marks impacts contraction expansion.
        In general, the impact of removal of special characters needs to be carefully investigated.
        Returns: reference to self object for continued chaining of properties 
        
        For reference, review these pages in the course from Module 1:
        - Overview of Regular Expressions
        - Practice Using Simple Expressions'''
        
        # YOUR CODE HERE 
        
        
        
        self.LsWords = [re.sub(r'[^\w\s]', '', word) for word in self.LsWords]
        
        # YOUR CODE HERE
        # raise NotImplementedError()
        
        return self.AddStats('Remove non-word characters') # finally, we save basic stats after this preprocessing

        
    ### TASK 6: Remove Stop Words ###########################################
    @property
    def Stop(self) -> object:
        '''Remove stopwords self.LsWords and save back to self.LsWords.
            Iterate over elements of self.LsWords and throw away those, 
            which are in self.SsStopWords regardless of letter casing.
        Hint: lower-case words only when checking membership in self.SsStopWords
        Returns: reference to self object for continued chaining of properties 
        
        For reference, review these pages in the course from Module 1:
        - Removing Stop Words
        - Remove Stop Words from a Document'''
        
        # YOUR CODE HERE 
        
        
        
        self.LsWords = [word for word in self.LsWords if word.lower() not in self.SsStopWords]
        
        # YOUR CODE HERE
        # raise NotImplementedError()
        
        return self.AddStats('Remove stopwords') # finally, we save basic stats after this preprocessing


    ### TASK 7: Normalization ###########################################
    @property
    def Norm(self) -> object:
        '''Normalization of accented characters or diacritics. Each word in self.LsWords 
            needs to be deaccented using normalize(), encode() and decode() methods.
            The list of normalized words is then saved back to self.LsWords
        Returns: reference to self object for continued chaining of properties 
        
        For reference, review these pages in the course:
        - Working with Characters
        - Work with Characters to Standardize a Vocabulary'''
        
        # YOUR CODE HERE 
        
        
        
        self.LsWords = [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in self.LsWords]

        
        # YOUR CODE HERE
        # raise NotImplementedError()
        
        return self.AddStats('Normalize accented characters') # finally, we save basic stats after this preprocessing


    ### TASK 8: Expand Contractions ###########################################
    @property
    def Exp(self) -> object:
        '''Applies character expansion to self.LsWords and saves results back to self.LsWords.
        1. Space-concatenate all tokens in self.LsWords.
        2. Apply contractions.fix() method to the full string
        3. Use split() to parse the pre-processed string back to list of tokens
        Returns: reference to self object for continued chaining of properties 
        
        For reference, review these pages in the course:
        - Expanding Contractions
        - Modify and Add a Contraction Map'''
        
        # YOUR CODE HERE 
        
        
        
        self.LsWords = contractions.fix(' '.join(self.LsWords)).split()
        
        # YOUR CODE HERE
        # raise NotImplementedError()
        
        return self.AddStats('Contraction expansion') # finally, we save basic stats after this preprocessing

    
    ### TASK 9: Stem ###########################################
    @property
    def Stem(self) -> object:
        '''Porter Stemming of self.LsWords
            Iterate over self.LsWords and stem each word using stem() method of pso object
        Returns: reference to self object for continued chaining of properties 
        
        For reference, review these pages in the course:
        - Stemming and Lemmatization
        - Stem and Lemmatize a Document to Measure Vocabulary Quality'''
        
        pso = nltk.stem.PorterStemmer()       # instantiates Porter Stemmer object
        
        # YOUR CODE HERE 
        
        
        self.LsWords = [pso.stem(word) for word in self.LsWords]
        
        
        # YOUR CODE HERE
        # raise NotImplementedError()
        
        return self.AddStats('Stem') # finally, we save basic stats after this preprocessing


    ### TASK 10: Lemmatize ###########################################
    @property
    def Lem(self) -> object:
        '''Wordnet Lemmatization of self.LsWords
            Iterate over LTssWordTag and lemmatize each word using its 
            WordNet POS tag and wlo.lemmatize() method. The POS tagging and
            tag conversion (from NLTK to WordNet tags) have been already implemented.
            Caution: nltk.pos_tag() is designed to take sentence tokens, not large documents.
        Example: 
            wlo.lemmatize('ran','v') returns 'run', but 
            wlo.lemmatize('ran','n') returns 'ran' (unintentionally)
        Returns: reference to self object for continued chaining of properties 
        
        For reference, review these pages in the course:
        - Stemming and Lemmatization
        - Stem and Lemmatize a Document to Measure Vocabulary Quality'''
        
        wlo = nltk.stem.WordNetLemmatizer()   # instantiates WordNet Lemmatizer object
        WNTag = lambda t: t[0].lower() if t[0] in 'ARNV' else 'n'   # Converts NLTK POS Tag to WordNet POS Tag
        # Create a list of tuples of words & their WordNet POS tags, 
        #    i.e. 'a' for adjectives, 'r' for adverbs, 'v' for verbs, 'n' for nouns and all else 
        LTssWordTag = [(word, WNTag(tag)) for word, tag in nltk.pos_tag(self.LsWords)]
        
        # YOUR CODE HERE 
        
        self.LsWords = [wlo.lemmatize(word, tag) for word, tag in LTssWordTag]
        
        
        # YOUR CODE HERE
        # raise NotImplementedError()
        
        return self.AddStats('Lemmatize') # finally, we save basic stats after this preprocessing


    def AddStats(self, sTask='') -> object:
        '''Object's preprocessing methods call AddStats() to save 
            basic word counts resulting from the NLP task.
        Input: 
            sTask: string,a brief description of the task. Eg. 'Low', 'Stem', ...
        Returns: reference to self object '''
        # Append a row (sStep, nWords, nVocab, nCorrVocab) at the bottom of self.df, where
        #   nWords =     count of words in self.LsWords
        #   SsWords =    set of unique words from self.LsWords
        #   nVocab =     count of words in SsWords
        #   nCorrVocab = count of words in the intersection of SsWords and self.SsLex
        
        # Note: This function is complete as-is. No edits are needed for the assignment.
        # This function is also not necessary to reference in the your code.
        
        SsWords = {s for s in self.LsWords}
        self.df.loc[len(self.df)] = [sTask, len(self.LsWords), len(SsWords), len(SsWords.intersection(self.SsLex))]
        return self     # Finally, return reference to the object itself


# ## Pipeline Tests
# 
# Here is a set of tests that evaluate whether initialization of `Pipe` class was implemented correctly.
# 
# ### Task 1: Object Initialization Tests
# * Methods required: `__init__`
# * **Note:** The following tests may take some time to run.

# In[3]:


# RUN CELL TO TEST YOUR CODE
LsDoc = "We'rè fighting CÓVÍD-19 in 2020; ánd we've WON!".split()
SsLex = {"We'rè", 'fighting', 'CÓVÍD-19'}
SsStopWords = {'in', 'ánd'}
@run_unittest
class TestObjInitialization(unittest.TestCase):
    def test_00(self): eq (Pipe(LsDoc).LsWords[:3], ["We'rè", 'fighting', 'CÓVÍD-19'])
    def test_01(self): eq (Pipe(LsDoc).SsLex, set())
    def test_02(self): eq (Pipe(LsDoc).SsStopWords, set())
    def test_03(self): eq (Pipe(LsDoc, SsLex=SsLex).LsWords[:3], ["We'rè", 'fighting', 'CÓVÍD-19'])
    def test_04(self): eq (Pipe(LsDoc, SsLex=SsLex).SsLex, {'CÓVÍD-19', 'fighting', "We'rè"})
    def test_05(self): eq (Pipe(LsDoc, SsLex=SsLex).SsStopWords, set())
    def test_06(self): eq (sorted(Pipe(LsDoc, SsLex='nltk').SsLex)[:3], ['!', '$.027', '$.03'])
    def test_07(self): eq (Pipe(LsDoc, SsStopWords=SsStopWords).LsWords[:3], ["We'rè", 'fighting', 'CÓVÍD-19'])
    def test_08(self): eq (Pipe(LsDoc, SsStopWords=SsStopWords).SsLex, set())
    def test_09(self): eq (Pipe(LsDoc, SsStopWords=SsStopWords).SsStopWords, {'ánd', 'in'})
    def test_10(self): eq (sorted(Pipe(LsDoc, SsStopWords='nltk').SsStopWords)[:3], ['a', 'about', 'above'])


# ### Task 2: Output Test 
# 
# These tests evaluate returned result of the initialized `Pipe` object.
# 
# * Methods required: `__init__`, `Out()`

# In[4]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestOutput(unittest.TestCase):
    def test_00(self): npt.assert_equal (Pipe(LsDoc + ['  \t\n\r!']).Out(), "We'rè fighting CÓVÍD-19 in 2020; ánd we've WON! !")


# ### Task 3: Lowercase Test
# * Methods required: `__init__`, `Low()`, `Out()`

# In[5]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestLowercase(unittest.TestCase):
    def test_00(self): eq (Pipe(LsDoc).Low.Out(), "we'rè fighting cóvíd-19 in 2020; ánd we've won!")


# ### Task 4: Number Removal Test
# * Methods required: `__init__`, `NoNum()`, `Out()`

# In[6]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestNumRemoval(unittest.TestCase):
    def test_00(self): eq (Pipe(LsDoc).NoNum.Out(), "We'rè fighting CÓVÍD- in ; ánd we've WON!")


# ### Task 5: Word Filter Tests
# * Methods required: `__init__`, `Words()`, `NoNum()`, `Out()`

# In[7]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestWordFilter(unittest.TestCase):
    def test_00(self): eq (Pipe(LsDoc).Words.Out(), 'Werè fighting CÓVÍD19 in 2020 ánd weve WON')
    def test_01(self): eq (Pipe(LsDoc).Words.NoNum.Out(), 'Werè fighting CÓVÍD in ánd weve WON')


# ### Task 6: Stopword Removal Tests
# * Methods required: `__init__`, `Stop()`, `Out()`

# In[8]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestStopwordRemoval(unittest.TestCase):
    def test_00(self): eq (Pipe(LsDoc).Stop.Out(), "We'rè fighting CÓVÍD-19 in 2020; ánd we've WON!")
    def test_01(self): eq (Pipe(LsDoc, SsStopWords='nltk').Stop.Out(), "We'rè fighting CÓVÍD-19 2020; ánd we've WON!")
    def test_02(self): eq (Pipe(LsDoc, SsStopWords={'ánd'}).Stop.Out(), "We'rè fighting CÓVÍD-19 in 2020; we've WON!")


# ### Task 7: Character Normalization Tests
# * Methods required: `__init__`, `Norm()`, `Stop()`, `Out()`

# In[9]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestCharNormalization(unittest.TestCase):
    def test_00(self): eq (Pipe(LsDoc).Norm.Out(), "We're fighting COVID-19 in 2020; and we've WON!")
    def test_01(self): eq (Pipe(LsDoc, SsStopWords='nltk').Norm.Stop.Out(), "We're fighting COVID-19 2020; we've WON!")


# ### Task 8: Contraction Expansion Tests
# * Methods required: `__init__`, `Norm()`, `Exp()`, `Stop()`, `Out()`

# In[10]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestContractionExpansion(unittest.TestCase):
    def test_00(self): eq (Pipe(LsDoc).Exp.Out(), "We'rè fighting CÓVÍD-19 in 2020; ánd we have WON!")
    def test_01(self): eq (Pipe(LsDoc).Norm.Exp.Out(), 'We are fighting COVID-19 in 2020; and we have WON!')
    def test_02(self): eq (Pipe(LsDoc, SsStopWords='nltk').Norm.Exp.Stop.Out(), 'fighting COVID-19 2020; WON!')


# ### Task 9: Stemming Tests
# * Methods required: `__init__`, `Norm()`, `Exp()`, `Stem()`, `Words()`, `Out()`

# In[11]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestStemming(unittest.TestCase):
    def test_00(self): eq (Pipe(LsDoc).Stem.Out(), "we'rè fight cóvíd-19 in 2020; ánd we'v won!")
    def test_01(self): eq (Pipe(LsDoc).Norm.Stem.Out(), "we'r fight covid-19 in 2020; and we'v won!")
    def test_02(self): eq (Pipe(LsDoc).Norm.Exp.Stem.Out(), 'we are fight covid-19 in 2020; and we have won!')
    def test_03(self): eq (Pipe(LsDoc).Norm.Exp.Stem.Words.Out(), 'we are fight covid19 in 2020 and we have won')


# ### Task 10: Lemmatization Tests
# * Methods required: `__init__`, `Norm()`, `Exp()`, `Words()`, `Low()`, `Lem()`, `Stop()`, `NoNum()`, `Out()`

# In[12]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class TestLemmatization(unittest.TestCase):
    def test_00(self): eq (Pipe(LsDoc).Lem.Out(), "We'rè fight CÓVÍD-19 in 2020; ánd we've WON!")
    def test_01(self): eq (Pipe(LsDoc).Norm.Exp.Words.Low.Lem.Out(), 'we be fight covid19 in 2020 and we have win')
    def test_02(self): eq (Pipe(LsDoc, SsStopWords='nltk').Norm.Exp.Words.Low.Lem.Stop.Out(), 'fight covid19 2020 win')
    def test_03(self): eq (Pipe(LsDoc, SsStopWords='nltk').Norm.Exp.Words.Low.Lem.Stop.NoNum.Out(), 'fight covid win')


# ### Task 11: df Tests
# * Methods required: all

# In[13]:


# RUN CELL TO TEST YOUR CODE
@run_unittest
class Testdf(unittest.TestCase):
    def test_00(self): eq (Pipe(LsDoc).df.values.ravel().tolist(), ['Initialize', 8, 8, 0])
    def test_01(self): eq (Pipe(LsDoc, SsLex='nltk').df.tail(1).values.ravel().tolist()[1:], [8, 8, 3])  # Brown lexicon is used to match words in it
    def test_02(self): eq (Pipe(LsDoc).Norm.Exp.df.tail(1).values.ravel().tolist()[1:], [10, 10, 0])


# <hr>
# 
# # **Optional: Use Your Pipeline**
# 
# Congratulations, you have just built a powerful preprocessing pipeline! Now, you can put this powerful pipeline machine to use on a larger corpus. First, load a text from Gutenberg library and run it through the cleaning pipeline to transform it into a list of words, which might represent the core meaning of the text.

# In[ ]:


_ = nltk.download(['gutenberg'], quiet=True)
LsBookWords = list(nltk.corpus.gutenberg.words('bryant-stories.txt')) #[:1000]
sSampleText = nltk.corpus.gutenberg.raw('bryant-stories.txt')[:500] + '...\n'
print(sSampleText)


# You can apply a sequence of preprocessing steps and output a dataframe with statistics at different steps of the pipeline. From what you have learned in this course, you might already have some expectations about the effectiveness of each step. For example, English language texts are less likely to benefit from removal of accent marks. You might expect that stemming and lemmatization would have the most dramatic drop in unique word count, but words produced from stemming may not be found in a dictionary, such as the lexicon created from NLTK's Brown Corpus. If you change the order of the steps, the counts are likely to change as well. In particular, special character normalization may remove quotation marks, which are needed for the contraction expansion to identify and fix contractions.
#  

# In[ ]:


get_ipython().run_line_magic('time', "pp = Pipe(LsBookWords, SsStopWords='nltk', SsLex='nltk').Low.Norm.Exp.Words.Stem.Stop.NoNum")
pp.Out()[:500]
pp.df


# ## Further Exploration with Your Pipeline
# 
# Investigate the original and clean document. How would you change the pipeline order to have the fewest unique and total words, but to have greatest overlap with Brown lexicon? Can you think of any other cleaning steps that might be useful for this pipeline?

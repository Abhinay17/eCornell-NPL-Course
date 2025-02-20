#!/usr/bin/env python
# coding: utf-8

# # **Part One of the Course Project**
# 
# <span style="color:black">In this project, you will complete user-defined functions (UDFs) to construct **document-term matrices** ([DTM](https://en.wikipedia.org/wiki/Document-term_matrix)) and **term frequency–inverse document frequency** ([TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)) DTM matrices from a corpus of wine reviews. Then, you will manipulate the matrix data stored in `pandas` DataFrames to retrieve key statistics and information. Finally, you will use that information to answer several questions about the NLTK wine reviews corpus.
#      
# <span style="color:black">You will test your implementation on a toy document, `LsNLP`, and on the much larger NLTK corpus of wine reviews, `LsWines`. The code below loads the documents and performs basic preprocessing.
#     
# As you work through this course project, you may need to consult the [Pandas documentation](https://pandas.pydata.org/pandas-docs/stable/index.html) to explore Pandas methods to help you in implementing these UDFs. 
# <hr style="border-top: 2px solid #606366; background: transparent;">
# 

# # Setup
# 
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries and data sets you will need to complete this part of the course project. 

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS; IS.ast_node_interactivity = "all"
import numpy as np, re, nltk, pandas as pd, numpy.testing as npt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import unittest
from colorunittest import run_unittest
eq, aeq, feq = npt.assert_equal, npt.assert_almost_equal, pd.testing.assert_frame_equal


# In[2]:


_ = nltk.download(['webtext','punkt','stopwords'], quiet=True)  # download NLTK corpora and databases
LsStopwords = nltk.corpus.stopwords.words('english')  # List of string words
LLsWineCorpus = nltk.corpus.webtext.sents('wine.txt') # short messages about wines (list of list of words)


# In[3]:


# Returns concatenated stemmed words as a string sentence. Only keeps tokens of letters
slo = nltk.PorterStemmer()  # instantiate a stemmer object for trimming words using Porter algorithm
CleanText = lambda LsDoc=['NLP','is','fun','!']: ' '.join(slo.stem(s) for s in LsDoc if s.isalpha()) # keep stemmed letter words only

LsNLP = ['nlp is fun', 'I like it a ton', 'more nlp makes better nlp']

print('Examples of wine reviews before and after preprocessing:')
print(' ')
print('CleanText processing time:')
get_ipython().run_line_magic('time', 'LsWines = [CleanText(s) for s in LLsWineCorpus]  # stem all words in LsWines (and remove non-letter tokens)')
print(' ')
print('View original sentences:')
print([' '.join(s) for s in LLsWineCorpus[:3]])   # show original sentences
print(' ')
print('View processed sentences:')
print(LsWines[:3])                                   # show preprocessed sentences
print(' ')


# ## Task 1: Create a Count-Based Document–Term Matrix (DTM)
# 
# <span style="color:black"> In this task, you will complete the `GetDTM()` function, which should use Scikit-learn's (SKL) [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) object to fit and transform a list of string sentences into a DTM. 
#     
# This function needs to take the input as `LsSents`, a list of string sentences, and return the DTM as a Pandas DataFrame with the vocabulary words as column names (see [`get_feature_names()`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.get_feature_names) method) and the sentences as row names. Additionally, the parameters of `GetDTM` should be passed through to `CountVectorizer()` so you can retain control over the vectorizer through your UDF. Note, Do **not** enable case sensitivity on the `CountVectorizer`.
#     
# <span style="color:black"> Hint: Recall that the vectorizer tokenizes the given documents into words and outputs a [sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix) of type [`scipy.sparse.csr.csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html), which stores only non-zero locations and values as a list-like structure. This sparse matrix can be converted into a NumPy array using the [`toarray()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.toarray.html#scipy.sparse.csr_matrix.toarray) method.

# #### Note: Anywhere you see the code <code>raise NotImplementedError()</code>, you can comment that line out, delete it, or move it after your <code>return</code> statement. 

# In[4]:


def GetDTM(LsSents=LsNLP, stop_words=LsStopwords, min_df=1, max_df=1.0, max_features=None) -> pd.DataFrame:
    ''' GetDTM is a wrapper for CountVectorizer, but returns a DataFrame instead of a sparse matrix.
    LsSents: list of string sentences
    stop_words, min_df, max_df, max_features: see CountVectorizer's documentation
    Return: DTM as pandas DataFrame object with vectorizer's vocabulary for the column names 
    and LsSents for the row names.     '''
    vectorizer = CountVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df, max_features=max_features)
    dtm = vectorizer.fit_transform(LsSents)
    df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names(), index=LsSents)
    return df


# In[5]:


# TEST & AUTOGRADE CELL
@run_unittest
class test_GetDTM(unittest.TestCase):
    def test_00(self):    
        df1 = pd.DataFrame([[1,0,1,0],[0,1,0,1]], columns=['fun','like','nlp','ton'], index=['nlp is fun','I like it a ton'])
        feq(GetDTM(LsNLP[:2]), df1)
    def test_01(self):
        df2 = pd.DataFrame([[1,0],[0,1]], columns=['fun','like'], index=['nlp is fun','I like it a ton'])
        feq(GetDTM(LsNLP[:2], stop_words=['is','it','nlp','ton']), df2)
    def test_02(self):
        df2 = pd.DataFrame([[1,0],[0,1]], columns=['fun','like'], index=['nlp is fun','I like it a ton'])
        feq(GetDTM(LsNLP[:2], max_features=2), df2)
    def test_03(self): eq(GetDTM(LsNLP, min_df=2).sum().sum(), 3)
    def test_04(self): eq(GetDTM(LsNLP, max_df=1).sum().sum(), 5)
    def test_05(self): eq(GetDTM(LsNLP).sum().sum(), 8)
    def test_06(self): eq(list(GetDTM(LsNLP).columns), ['better', 'fun', 'like', 'makes', 'nlp', 'ton'])
    def test_07(self): eq(GetDTM(LsNLP).index[0], 'nlp is fun')
    def test_08(self): eq(GetDTM(LsWines).sum().sum(), 15637)
    def test_09(self): eq(GetDTM(LsWines, stop_words=[]).sum().sum(), 22653)
    def test_10(self): eq(GetDTM(LsWines, stop_words=LsStopwords+['wine','veri','thi','good','fruit']).sum().sum(), 14005)
    def test_11(self): eq(GetDTM(LsWines, min_df=300).sum().sum(), 1054)
    def test_12(self): eq(GetDTM(LsWines, min_df=300, max_df=310).sum().sum(), 311)
    def test_13(self): eq(list(GetDTM(LsWines, min_df=300, max_df=310, max_features=2).columns)[0], 'fruit')


# ## Task 2: Address Sparsity in the Count-Based DTM
# 
# <span style='color:black'>Here, you will complete the `GetFrac()` UDF so that it computes the fraction of `nValue` values in the DTM DataFrame as a way to calculate sparsity. Recall that real-world corpora typically have high sparsity. Do you also observe high sparsity in the wine reviews corpus?
# 

# In[6]:


def GetFrac(dfDTM=GetDTM(), nValue=0) -> float:
    '''Return the fraction of nValue elements in dfDTM dataframe
    Compute how many nValue elements are in the dataframe 
    (as proportion of the total number of elements).     '''
    df = dfDTM.copy()  # make a copy of dataframe to avoid modifying original values on a reference
    total_elements = df.size
    count_nValue = (df == nValue).sum().sum()
    return count_nValue / total_elements


# In[7]:


# TEST & AUTOGRADE CELL
try:    dfDTM_Wines, dfDTM_NLP = GetDTM(LsWines), GetDTM(LsNLP)
except: dfDTM_Wines = dfDTM_NLP = None # If not implemented, save None value
    
@run_unittest
class test_GetFrac(unittest.TestCase):
    def test_00(self): aeq(GetFrac(dfDTM_NLP, nValue=0), 0.611111111111111, 4)
    def test_01(self): aeq(GetFrac(dfDTM_NLP, nValue=1), 0.3333333333333333, 4)
    def test_02(self): aeq(GetFrac(dfDTM_Wines, nValue=0), 0.9975999585059148, 4)
    def test_03(self): aeq(GetFrac(dfDTM_Wines, nValue=1), 0.0023728653529357688, 4)


# ## Additional Demos of Helpful Pandas Methods
# 
# For some of the functions that follow, you you may find it convenient to perform all operations on the input DataFrame by calling appropriate methods. Methods to consider (some of which you have seen before in the practice exercises) include `reset_index()`, `sort_values()`, `values`, `sum()`, `max()`, `min()`, `replace()`, and `tolist()`. The cells below contain some simple examples to illustrate how these work to help you understand where they might be useful in your own code.
# 
# First, create a DataFrame to practice on using the `GetDTM()` function you coded above, accepting all the defaults:

# In[8]:


df_practice = GetDTM()
df_practice


# The `sum()` method is straightforward and adds up the values of each column by default and returns it as a series, with the column labels serving as the index.

# In[9]:


df_practice.sum()


# You can change the default behavior of sum by changing the axis over which to perform the sum. 

# In[10]:


df_practice.sum(axis=1)


# If you want to know the largest value in each column instead of the sum, the `max()` method can be helpful.

# In[11]:


df_practice.max()


# Conversely, if you want to know the smallest values in each column, try the `min()` method.

# In[12]:


df_practice.min()


# Sometimes when working with sparse objects, you will want to replace zeros with NaN. This can be done using the `replace()` method and the NumPy `nan` function.

# In[13]:


df_practice.min().replace(0,np.nan)


# You can stack methods to produce useful outputs. For example, after calculating the sum of the word values, it may be helpful to sort those values. By default, the `sort_values` method will do this in ascending order:

# In[14]:


df_practice.sum().sort_values()


# You can generate the list in descending order (which can be useful when you want to find the highest values) by specifying `ascending=False` within the `sort_values()` method:

# In[15]:


df_practice.sum().sort_values(ascending=False)


# If you just want the values from the DataFrame, Pandas provides the `values` method which returns all values in the DataFrame as a list (or if dealing with multiple columns, a list of lists) embedded in an array.

# In[16]:


df_practice.sum().sort_values(ascending=False).values


# The `tolist()` method reformats the array into a list (or list of lists).

# In[17]:


df_practice.sum().sort_values(ascending=False).values.tolist()


# Perhaps the most challenging method in the list above is the `reset_index()` method. Take a look at an example of what happens when you use the `sum()` method, and then compare that to the output after calling `reset_index()`:

# In[18]:


df_practice.sum()


# In[19]:


df_practice.sum().reset_index()


# The `reset_index()` method has taken the series index produced by the `sum()` method and made it into a column labeled "index," taken the summed values and put them into a column labeled "0," and replaced the actual index with a range from 0 to the end of the values.
# 
# You can verify by using the Pandas `columns` method that this process creates two columns, which can now be independently acted upon.

# In[20]:


df_practice.sum().reset_index().columns


# In[21]:


df_practice.sum().reset_index()['index']


# In[22]:


df_practice.sum().reset_index()[0]


# Note in the example above, subsetting by 0 returned the values in the column labeled "0," not the first column in the DataFrame, which was labeled "index." This context switch can take some getting used to. When in doubt, don't forget to print intermediate outputs to ensure you are working on the right section of the data.

# Finally, when sorting DataFrames that have more than one column, you must specify the sort order when calling `sort_values` using the `by` argument and the column or order of columns you want to sort by. Take a look at the examples below:

# In[23]:


# You can uncomment the line of code below to see the error message when you don't specify the column to sort by:

#df_practice.sum().reset_index().sort_values()


# In[24]:


# Sort by the 'index' column

df_practice.sum().reset_index().sort_values(by = 'index')


# In[25]:


# Sort by the 0 column

df_practice.sum().reset_index().sort_values(by = 0)


# In[26]:


# Sort by 0, then by 'index' as applicable

df_practice.sum().reset_index().sort_values(by = [0, 'index'])


# Note that the previous two examples returned the same output, because the initial sort for "index" was ascending alphabetical order. You can change this behavior by specifiying whether ascending is True or False for each column specified.

# In[27]:


df_practice.sum().reset_index().sort_values(by = [0, 'index'], ascending=[True,False])


# The examples above are often not the only way to accomplish a given task, and they do not represent the complete set of programming skills needed to complete all of the functions below. Please refer to the practice exercises and relevant documentation or equivalent resources for additional information. If you have questions that cannot be answered with these resources, please reach out to your course facilitator.

# ## Task 3: Identify the Most Common Words in the DTM Overall
# 
# <span style="color:black">Next, you'll complete a function `MostCommonWords1` that computes each column's total and outputs only the `n` most frequent words in DTM. In case of a tie, your function should order words alphabetically. 
#     
# For example, consider the following DTM and `n=3`:
# 
# |.|better|fun|makes|nlp|
# |-|-|-|-|-|
# |more nlp makes better nlp|1|0|1|2|
# |better nlp is fun|1|1|0|1|
# 
# The final output should be the following:
# 
#     [['nlp', 3], ['better', 2], ['fun', 1]]
#     
# Since "fun" is alphabetically higher than "makes," it's returned as the third most frequent word, even though their actual counts are the same.

# In[28]:


def MostCommonWords1(dfDTM=GetDTM(), n=5) -> [[str, int], ...]:
    ''' Return top n most frequent words with their counts as a list of lists.
    If frequencies are tied, order terms alphabetically.'''
    df = dfDTM.copy()  # make a copy of dataframe to avoid modifying original values on a reference
    word_counts = df.sum(axis=0)
    sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    top_n_words = sorted_words[:n]
    return [[word, count] for word, count in top_n_words]


# In[29]:


# TEST & AUTOGRADE CELL
@run_unittest
class test_MostCommonWords1(unittest.TestCase):
    def test_00(self): eq(MostCommonWords1(dfDTM_NLP, n=5), [['nlp', 3], ['better', 1], ['fun', 1], ['like', 1], ['makes', 1]])
    def test_01(self): eq(MostCommonWords1(dfDTM_Wines, n=5), [['veri', 380], ['good', 363], ['fruit', 311], ['quit', 303], ['thi', 289]])
    def test_02(self): eq(MostCommonWords1(GetDTM(LsWines[:100]), n=5), [['good', 15], ['bit', 11], ['thi', 11], ['wine', 11], ['dri', 9]])


# ## Task 4: Identify the Most Common Words by Occurrence in Separate Documents
# 
# Next, you'll complete the function `MostCommonWords2`, which counts only the number of sentences (documents) in which the word appears, regardless of its count within a sentence. An easy way to achieve this is to convert all non-zero values to 1 (on a copy of `dfDTM`!) and then call the function you completed in Task 3, `MostCommonWords1()`.
# 
# For example, consider the following DTM and `n=3`:
# 
# |.|better|fun|makes|nlp|
# |-|-|-|-|-|
# |more nlp makes better nlp|1|0|1|2|
# |better nlp is fun|1|1|0|1|
# 
# Then the output should be the following:
# 
#     [['better', 2], ['nlp', 2], ['fun', 1]]
#     
# Since alphabetically "better" $>$ "nlp" and "fun" $>$ "makes," "better" and "nlp" appear in two sentences, while "fun" and "makes" each appears in one sentence.

# In[30]:


def MostCommonWords2(dfDTM=GetDTM(), n=5) -> [[str, int], ...]:
    ''' Return top n most frequent words with their counts.
    Each word is counted only once per sentence. If frequencies are tied, order terms alphabetically.'''
    df = dfDTM.copy()  # make a copy of dataframe to avoid modifying original values on a reference
    df[df > 0] = 1
    
    # Sum across the rows to get the count of sentences (documents) each word appears in
    word_counts = df.sum(axis=0)
    
    # Use the MostCommonWords1 function to sort by frequency and alphabetically
    return MostCommonWords1(dfDTM=df, n=n)


# In[31]:


# TEST & AUTOGRADE CELL
@run_unittest
class test_MostCommonWords2(unittest.TestCase): # verify input dataframes and outputs of MostCommonWords2()
    def test_00(self): eq(type(dfDTM_NLP), pd.DataFrame)
    def test_01(self): eq(list(dfDTM_NLP.columns), ['better', 'fun', 'like', 'makes', 'nlp', 'ton'])
    def test_02(self): eq(list(dfDTM_NLP.index), ['nlp is fun', 'I like it a ton', 'more nlp makes better nlp'])
    def test_03(self): eq(dfDTM_NLP.shape, (3,6))
    def test_04(self): eq(dfDTM_NLP.values, [[0, 1, 0, 0, 1, 0],  [0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 2, 0]])
    def test_05(self): eq(MostCommonWords2(dfDTM_NLP, n=5), [['nlp', 2], ['better', 1], ['fun', 1], ['like', 1], ['makes', 1]])
    
    def test_06(self): eq(type(dfDTM_Wines), pd.DataFrame)
    def test_07(self): eq(list(dfDTM_Wines.columns[[0,1,2,-3,-2,-1]]), ['abandon', 'ablout', 'abov', 'zing', 'zingi', 'zoo'])
    def test_08(self): eq(list(dfDTM_Wines.index[[0,1,-2,-1]]), ['love delic fragrant rhone wine', 'polish leather and strawberri',
       'i feel thi hasn t the fine to be great but it is veri good', 'ul'])
    def test_09(self): eq(dfDTM_Wines.shape, (2984, 2158))
    def test_10(self): eq(dfDTM_Wines.sum().sum(), 15637)
    def test_11(self): eq(MostCommonWords2(dfDTM_Wines, n=5), [['veri', 366], ['good', 353], ['fruit', 305], ['quit', 297], ['wine', 281]])


# ## Task 5: Identify Most Frequent Words in DTM
# 
# Complete the function `MostCommonWords3` so that it returns $n$ the most frequent words across all documents. In case of a count tie, return them alphabetically. The output is a list of lists in the form `[str, int]` with a word and its count in DTM.
# 
# For example, consider the following DTM and `n=4`:
# 
# |.|better|fun|makes|nlp|
# |-|-|-|-|-|
# |more nlp makes better nlp|1|0|1|2|
# |better nlp is fun|1|1|0|1|
# 
# Then the output should be the following:
# 
#     [['nlp', 2], ['better', 1], ['fun', 1], ['makes', 1]]
#     
# Since "better" is alphabetically higher than "fun," it's returned first.

# In[32]:


def MostCommonWords3(dfDTM=GetDTM(), n=5) -> [[str, int], ...]:
    ''' Return top n most frequent words from the sentences in DTM.
    Basically, a word with the highest count value in DTM is the most common. 
    Then word with the second highest count value in DTM is the second most common. And so on.
    If frequencies are tied, order terms alphabetically.
    Each word should be returned once.
    Hint: you can find max value from each column and order by (max, index name), 
        then keep only top n words.
    Inputs: 
        dfDTM: DTM dataframe with column words and row sentences. Values as frequency counts.
        n: number of words to return
    Ouput: a list of lists in the form [[word, largest count 1],...,[word, smallest count n])   '''
    df = dfDTM.copy()  # make a copy of dataframe to avoid modifying original values on a reference
    out = []           # deired (list) structure of the output
    word_counts = df.max(axis=0)  # Get max frequency for each word
    word_df = pd.DataFrame(word_counts).reset_index()
    word_df.columns = ['word', 'count']
    word_df = word_df.sort_values(by=['count', 'word'], ascending=[False, True])
    top_n_words = word_df.head(n)
    out = top_n_words.values.tolist()
    return out


# In[33]:


# TEST & AUTOGRADE CELL
df1 = pd.DataFrame([[1,0,1,2],[1,1,0,1]], columns=['better','fun','makes','nlp'])
@run_unittest
class test_MostCommonWords3(unittest.TestCase):
    def test_00(self): eq(type(MostCommonWords3(df1)), list)
    def test_01(self): eq(MostCommonWords3(df1), [['nlp', 2], ['better', 1], ['fun', 1], ['makes', 1]])
    def test_02(self): eq(MostCommonWords3(dfDTM_NLP, n=5), [['nlp', 2], ['better', 1], ['fun', 1], ['like', 1], ['makes', 1]])
    def test_03(self): eq(MostCommonWords3(dfDTM_Wines, n=5), [['depth', 3], ['fruit', 3], ['thi', 3], ['top', 3], ['veri', 3]])
    def test_04(self): eq(MostCommonWords3(dfDTM_Wines.iloc[:,:1000], n=5), [['depth', 3], ['fruit', 3], ['accord', 2], ['acid', 2], ['bay', 2]])


# ## Task 6: Find Documents With the Most Occurrences of a Word
# 
# Complete the UDF `SentWithMostDups` that, given the word `sWord`, retrieves the sentence (stored in `dfDTM`'s row index), that contains the most occurrences of `sWord`. If multiple sentences contain the same number of occurrences of `sWord`, sort alphabetically and return only the first sentence.

# In[34]:


def SentWithMostDups(dfDTM=GetDTM(), sWord='depth') -> str:
    '''Returns the sentence containing most instances of sWord.  Sentences are stored as indices of dfDTM.
    If word is not in the columns of dfDTM, return None.
    In case of ties, order alphabetically and retrieve the first sentence.'''
    df = dfDTM.copy()  # make a copy of dataframe to avoid modifying original values on a reference
    if sWord not in df.columns:
        return None
    word_counts = df[sWord]
    max_count = word_counts.max()
    max_sentences = word_counts[word_counts == max_count].index.tolist()
    return sorted(max_sentences)[0]


# In[35]:


# TEST & AUTOGRADE CELL
@run_unittest
class test_SentWithMostDups(unittest.TestCase):
    def test_00(self): eq(SentWithMostDups(dfDTM_NLP, sWord='NLP'), None)
    def test_01(self): eq(SentWithMostDups(dfDTM_NLP, sWord='nlp'), 'more nlp makes better nlp')
    def test_02(self): eq(SentWithMostDups(dfDTM_Wines, sWord='depth'), 'not rate depth depth and more depth')
    def test_03(self): eq(SentWithMostDups(dfDTM_Wines, sWord='wine'), 'a good big scale wine top chianti as i d expect from thi wine in such a fine vintag')


# ## Task 7: Create a TF-IDF DTM From the Count-Based DTM
# 
# <span style="color:black">Complete the `GetTFIDF()` function so that it transforms the DTM DataFrame into a TF-IDF DTM that contains fractional weights rather than count frequencies. The new DataFrame should retain the dimensions, indices, and column names of the count-based DTM you created. Recall that in a TF–IDF matrix, values are closer to 1 for more important words that appear either less frequently or are more concentrated in a single document and closer to zero for words that are broadly spread out in large counts.

# In[36]:


def GetTFIDF(dfDTM=GetDTM(), use_idf=True, smooth_idf=True) -> pd.DataFrame:
    '''Return TF-IDF dataframe of the same dimensions as dfDTM and with the same index and column names.
    use_idf, smooth_idf: see help for  TfidfTransformer(). Pass these parameters to TfidfTransformer. '''
    df = dfDTM.copy()  # make a copy of dataframe to avoid modifying original values on a reference
    transformer = TfidfTransformer(use_idf=use_idf, smooth_idf=smooth_idf)
    tfidf = transformer.fit_transform(df)
    df_tfidf = pd.DataFrame(tfidf.toarray(), index=df.index, columns=df.columns)
    return df_tfidf


# In[37]:


# TEST & AUTOGRADE CELL
@run_unittest
class test_GetTFIDF(unittest.TestCase):
    def test_00(self): aeq(GetTFIDF(dfDTM_NLP).sum().sum(), 4.5108460178771494, 4)
    def test_01(self): aeq(GetTFIDF(dfDTM_Wines).sum().sum(), 6194.966362333746, 4)
    def test_02(self): aeq(GetTFIDF(dfDTM_Wines, smooth_idf=False).sum().sum(), 6175.555391211886, 4)

dfTFIDF_NLP = GetTFIDF(dfDTM_NLP)
dfTFIDF_Wines = GetTFIDF(dfDTM_Wines)


# ## Task 8: Identify the Most Important Words in the Corpus
# 
# <span style="color:black">With the information in the TF-IDF DTM, you can automatically compile a list of stopwords that are too generic to be useful for your specific corpus of interest. For example, in this wine reviews corpus the word "wine" is likely very common. In this task, you will complete the `MostImportantWords()` UDF so that it returns words with the highest TF-IDF weights. Use each column's max value to measure the peak importance of each word. Then return the words ordered by this peak importance in decreasing order. In the case of a tie in weights, order words alphabetically.

# In[38]:


def MostImportantWords(dfDTM, n=5) -> [[str, float]]:
    df = dfDTM.copy()
    max_values = df.max(axis=0)  # Get the max value of each word (column)
    words_and_values = list(zip(max_values.index, max_values.values))
    words_and_values.sort(key=lambda x: (-x[1], x[0]))
    return [[word, round(value, 4)] for word, value in words_and_values[:n]]


# In[39]:


# TEST & AUTOGRADE CELL
@run_unittest
class test_MostImportantWords(unittest.TestCase):
    def test_00(self): eq(MostImportantWords(dfTFIDF_NLP.round(4), n=3), [['fun', 0.796], ['nlp', 0.7324], ['like', 0.7071]])
    def test_01(self): eq(MostImportantWords(dfTFIDF_Wines, n=5), [['anise', 1.0], ['appet', 1.0], ['attract', 1.0], ['auster', 1.0], ['bad', 1.0]])
    def test_02(self): eq(MostImportantWords(dfTFIDF_Wines[:10].round(4), n=3), [['rough', 1.0], ['liquoric', 0.7153], ['uninspir', 0.6959]])


# ## Task 9: Identify Auto Stopwords
# 
# <span style="color:black">In this final task, you will complete the UDF `LeastImportantWords` so it uses the TF-IDF DTM to automatically identify stopwords in a given corpus of documents. To do so, you will need to find the minimum non-zero value of each column. 
#     
# <span style="color:black">Hint: You can order the words by their increasing lowest weight, resolving the ties alphabetically as before. To ignore zeros in the `min()` method of a dataframe, you can use the `replace()` method to replace zeros with `np.nan` values, which are ignored by aggregating methods.

# In[40]:


def LeastImportantWords(dfDTM, n=5) -> [[str, int], ...]:
    '''Return auto-detected stop words. These are identified by the lowest non-zero weights in their columns.
    Order auto-stopwords in weight-increasing order, resolving ties with alphabetical ordering of the words. '''
    df = dfDTM.copy()  # make a copy of dataframe to avoid modifying original values on a reference
    df_replace_zeros = df.replace(0, np.nan)  # Replace zeros with NaN to ignore them in min calculation
    min_values = df_replace_zeros.min(axis=0)  # Find minimum value for each column
    words_and_values = list(zip(min_values.index, min_values.values))
    words_and_values.sort(key=lambda x: (x[1], x[0]))
    return [[word, round(value, 4)] for word, value in words_and_values[:n]]


# In[41]:


# TEST & AUTOGRADE CELL
@run_unittest
class test_LeastImportantWords(unittest.TestCase):
    def test_00(self): eq(LeastImportantWords(dfTFIDF_Wines.round(4), n=5), [['wine', 0.0865], ['thi', 0.0881], ['veri', 0.0909], ['fruit', 0.0953], ['top', 0.1056]])
    def test_02(self): eq(LeastImportantWords(dfTFIDF_NLP.round(4), n=5), [['better', 0.4815], ['makes', 0.4815], ['nlp', 0.6053], ['like', 0.7071], ['ton', 0.7071]])
    def test_03(self): eq(LeastImportantWords(GetTFIDF(GetDTM(LsWines, stop_words=[])).round(4), n=5), [['and', 0.0621], ['but', 0.0675], ['of', 0.069], ['the', 0.0723], ['it', 0.0764]])


# In[ ]:





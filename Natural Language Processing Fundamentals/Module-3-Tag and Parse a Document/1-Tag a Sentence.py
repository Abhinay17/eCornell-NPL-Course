#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Clear the Python environment of any previously loaded variables, functions, and libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import nltk, pandas as pd, collections


# You will use tools from the `nltk` library to perform parts of speech (POS) tagging on a famous Walt Disney phrase. Some of these tools will require additional datasets and copora, which you can download to local storage with `nltk.download()`.

# In[2]:


wlo = nltk.stem.WordNetLemmatizer()
nltk.download(['universal_tagset','wordnet','punkt','averaged_perceptron_tagger','tagsets'], quiet=True)


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# ## POS Tagging
# 
# You will use the `nltk.pos_tag()` method, which accepts a list of word strings from a parsed sentence and returns a list of tuples of the form: (word, POS tag). To begin, apply a word tokenizer on a phrase to get the list of word tokens from a famous quote by Walt Disney. Note that if you are performing POS tagging on a document, you should tokenize the document into sentences first because tagging quality is best on single sentences, rather than short phrases or longer paragraphs. 

# In[3]:


sDoc = "The way to get started is to quit talking and begin doing." # Walt Disney's quote
LsWords = nltk.word_tokenize(sDoc)
print(LsWords)


# Now that you have a list of word tokens (from a parsed sentence), you can apply the POS tagger `nltk.pos_tag` to generate a list of corresponding tags based on the UPenn tagset, which you'll examine in more detail later in this activity.

# In[4]:


LTsWordTags = nltk.pos_tag(LsWords)
print(LTsWordTags)


# Examine the output, which is a list of tuples in the form (word, POS tag).
# 
# To evaluate how well your tagging method has worked, you can examine the distribution of tags in a sentence. Similar to the `Counter()` class from Module 1, `nltk.FreqDist()` counts elements within a container (such as a list or a tuple). By applying this to the tags stored in the tuples and using its `most_common()` method, you can get the list of most common tags sorted by their counts. Note that we count the tags stored in tuples, not the tuples (which are likely to be unique). NLTK library has a similar counter accessible through `nltk.FreqDist()`. 

# In[5]:


print(nltk.FreqDist(tag for word, tag in LTsWordTags).most_common())


# There are many tag sets, each one with its advantages and disadvantages. The `nltk.pos_tag()` returns UPenn tag set ([from University of Pennsylvania's Treebank Project](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)). You can read information about each tag by using `nltk`'s `help` attribute.
# 
# You can use `upenn_tagset()` method to learn more about individual tags or view all available tags.

# In[6]:


nltk.help.upenn_tagset('T')


# <div id="blank_space" style="padding-top:20px">
#     <details>
#         <summary>
#             <div id="button" style="color:white;background-color:#de2424;padding:10px;border:3px solid #B31B1B;border-radius:30px;width:230px;text-align:center;float:left;margin-top:-15px"> 
#                 <b>ABOUT UPENN TAG SET → </b>
#             </div>
#         </summary>
#         <div id="button_info" style="padding:20px;background-color:#eee;border:3px solid #aaa;border-radius:30px;margin-left:25px;">
#             <p style="padding:15px 2px 2px 2px">
#            <p>In English, a word in a sentence can assume different functions, which often depends on the sentence's structure itself. For example, "building" is a verb in "building a house" and a noun in "a tall building". In the phrase "I drive to work", it's not even clear whether "work" is a noun or a verb. This ambiguity might disappear with a larger context in a phrase "I drive to work, where we share ideas." The examples highlight the challenges of automatic tagging.</p>
# <p>Among many POS tag sets, the most common (and the default in <code>nltk</code>) is <a class="inline_disabled" href="https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html" target="_blank" rel="noopener">Penn tag set</a> from a <a class="inline_disabled" href="https://www.cis.lmu.de/~schmid/tools/TreeTagger/data/Penn-Treebank-Tagset.pdf" target="_blank" rel="noopener">Penn Treebank Project</a> developed by <a class="inline_disabled" href="https://www.ling.upenn.edu/people/santorini" target="_blank" rel="noopener">Beatrice Santorini</a> at University of Pennsylvania. It contains 36 core tags plus a few more auxiliary tags. Tags are somewhat hierarchical with 2-letter tags branching to 3- and 4-letter subtags. You can review the full set of tags with <code>nltk.help.upenn_tagset()</code> command, but here are a few frequent tags we will see in the course:</p>
# <ul>
#     <li><strong>NN</strong>: singular noun <em>(cat, fox, syllabus)</em>
#         <ul>
#             <li><strong>NNS</strong>: plural (<em>cats</em>, <em>foxes,&nbsp;</em><i>syllabi</i>)</li>
#             <li><strong>NNP</strong>: proper singular (<em>Marie</em>, <em>Aditya, Victoria</em>)
#                 <ul>
#                     <li><strong>NNPS</strong>: proper plural <em>(Cornellians, New Yorkers, (The) Kennedys, Russians)</em></li>
#                 </ul>
#             </li>
#         </ul>
#     </li>
#     <li><strong>VB</strong>: base verb form <em>(learn, give, do)</em>
#         <ul>
#             <li><strong>VBD:</strong> past tense (<em>learnt, gave, did</em>)</li>
#             <li><strong>VBG:</strong> gerund or present participle (<em>learning, giving, doing</em>)</li>
#             <li><strong>VBN:</strong> past participle (<em>learnt, given, done</em>)</li>
#             <li><strong>VBP:</strong> present tense, singular, non-3rd person (<em>I learn, I give, I do</em>)</li>
#             <li><strong>VBZ:</strong> present tense, singular, 3rd person (<em>(he) works, (she) learns</em>)</li>
#         </ul>
#     </li>
#     <li><strong>JJ:</strong> adjective, numeral, ordinal (<em>third, pre-war, multilingual</em>)
#         <ul>
#             <li><strong>JJR:</strong> adjective, comparative (<em>cheaper, busier, cleaner</em>)</li>
#             <li><strong>JJS:</strong> adjective, superlative (<em>cheapest, busiest, cleanest</em>)</li>
#         </ul>
#     </li>
#     <li><strong>.</strong> : sentence terminator</li>
# </ul>
# <p>This list is so comprehensive that even punctuation symbols are issued their own Penn POS tags. <p/>
#         </div>

#   
# ## Lemmatize Words by Converting Between Tagsets
# 
# <span style="color:black">To properly lemmatize words, you will use a different tagset, called the WordNet tagset. To get this tagset, you can use the following user-defined function to modify the UPenn tag set to include only the four most basic categories of languages: adjectives (`'a'`), verbs (`'v'`), nouns (`'n'`), and adverbs (`'r'`).

# In[7]:


def get_wordnet_tag(tag):
    if tag.startswith('J'):   return 'a'  # adjective
    elif tag.startswith('V'): return 'v'  # verb
    elif tag.startswith('N'): return 'n'  # noun
    elif tag.startswith('R'): return 'r'  # adverb
    else: return 'n'


# Compare the default lemmatization (which tags all words as nouns) with and without the WordNet POS tagset. 

# In[8]:


print([wlo.lemmatize(w) for w in LsWords])  ## lemmatize without POS tagging
print([wlo.lemmatize(w, get_wordnet_tag(t)) for w,t in LTsWordTags])  ## Lemmatize after POS tagging with the Wordnet tagset


# Notice that, with the WordNet tagset, more words are correctly replaced with their proper root forms. For example, `'started'` is converted to `'start`' and `'is'` is converted to `'be'`. 
#     
# ## Comparing Different Tagsets
# 
# Different tagsets have different strengths. [Universal tagset](https://universaldependencies.org/u/pos/) is another common tagset that is easier to read but more rigid than the UPenn tagset, and more flexible than the WordNet tagset. Penn offers a finer set of POS tags. For example, noun tags can be plural, verb tags can represent some tenses, etc. A more comprehensive list is shown below. 

# In[ ]:


sTxt = "In 2020, the quick brown fox jumped over the lazy dog's puppies..."

LsWords = nltk.word_tokenize(sTxt)
LTsPT = nltk.pos_tag(LsWords, tagset=None) # UPenn tags
LTsWT = [(w, get_wordnet_tag(t)) for w, t in LTsPT] # WordNet tags
LTsUT = nltk.pos_tag(LsWords, tagset='universal') # universal tags

dfPT = pd.DataFrame(LTsPT, columns=['word','Penn'])
dfWT = pd.DataFrame(LTsWT, columns=['word','WordNet'])
dfUT = pd.DataFrame(LTsUT, columns=['word','Universal'])

dfTags = pd.concat([dfPT, dfWT.WordNet, dfUT.Universal], axis=1).set_index('word')
dfTags.T


# Notice that almost all of the WordNet tags are `'n'` (nouns). The Universal tagset distinguishes these further into `'ADP'` (adposition), determiners, and nouns.
# 
# ## Table of UPenn Tags
# 
# If you're ever unsure of what a particular tag means, you can find documentation, examples, and definitions right in `nltk`. The following table displays the documentation nicely.

# In[ ]:


pd.set_option('max_rows', 100, 'display.max_colwidth', 0)
DTagSet = nltk.data.load('help/tagsets/upenn_tagset.pickle')  # dictionary of POS tags
pd.DataFrame(DTagSet, index=['Definition', 'Examples']).T.sort_index().reset_index().rename(columns={'index':'Tag'})


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Optional Practice**
# 
# Load several famous quotes and text from "[Alice's Adventures in Wonderland](https://en.wikipedia.org/wiki/Knave_of_Hearts_(Alice%27s_Adventures_in_Wonderland))" by Lewis Carroll. The text is loaded in two different ways: first, as parsed sentences of parsed words and, second, as all parsed words. 
#    

# In[ ]:


sQuote1 = "The only true wisdom is in knowing you know nothing." # a quote from Socrates
sQuote2 = "Learn from yesterday, live for today, hope for tomorrow. The important thing is not to stop questioning." # a quote from Albert Einstein

_ = nltk.download(['gutenberg'], quiet=True)   # download Gutenberg corpus
LLsWords = nltk.corpus.gutenberg.sents('carroll-alice.txt') # list of lists of words (i.e. sentences of words)
LsWords = nltk.corpus.gutenberg.words('carroll-alice.txt')  # list of words (i.e.flattened sentences)
print(LLsWords[:2])
print(LsWords[:10])


# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See <b>solution</b> drop-down to view the answer.
# 
# ## Task 1
# 
# Tokenize the sentence string, `sQuote1`, into word tokens and identify their POS tags. Save the resulting list of tuples `\[(word, POS tag)\]` to the `LTsWordTags1` variable.
# 
#  <b>Hint:</b> Try <code>nltk.pos_tag()</code> and <code>nltk.word_tokenize()</code> functions applied to the quote.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#      <pre>   
# LTsWordTags1 = nltk.pos_tag(nltk.word_tokenize(sQuote1))
# print(LTsWordTags1)
#         </pre>
#         </details> 
# </font>
# <hr>

# ## Task 2
# 
# Count the frequencies of all POS tags in `LTsWordTags1`.
# 
# <b>Hint:</b> Frequency counting can be easily done with either <code>nltk.FreqDist()</code> or <code>collections.Counter()</code>. You will need to extract just the tags from the list of tuples.

# In[ ]:


# check solution here


# 
# 
# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#      <pre>   
# print(nltk.FreqDist(tag for word, tag in LTsWordTags1).most_common())
#         </pre>
#         </details> 
# </font>
# <hr>

# ## Task 3
# 
# Lemmatize words in `LTsWordTags1` using the appropriate WordNet tags. Then join all lemmatized words into a sentence.
# 
# <b>Hint:</b> Try <code>str.join()</code> method on the list of words lemmatized with <code>wlo.lemmatize()</code>.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# print(sQuote1)
# print(' '.join([wlo.lemmatize(w, get_wordnet_tag(t)) for w,t in LTsWordTags1]))
#         </pre>
#         </details>
# </font>
# <hr>

# ## Task 4
# 
# Complete the user-defined function (UDF), `Sent2Lemmas()`, which takes a list of word tokens from a single sentence and lemmatizes them using the correct WordNet tags. The output of the function with default arguments should be:
# 
#         >>> Sent2Lemmas()
#         ['I', 'be', 'love', 'NLP', '.']
# 
# Then evaluate `Sent2Lemmas()` at the word-tokenized `sQuote1`.
# 
# <b>Hint:</b> You will need to use <code>get_wordnet_tag()</code> tag-mapping function we created above.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# def Sent2Lemmas(LsSent=['I', 'am', 'loving', 'NLP', '.']):
#     '''sSent: list of word tokens from a single sentence
#     Return: a list of lemmatized words'''
#     return [wlo.lemmatize(w, get_wordnet_tag(t)) for w,t in nltk.pos_tag(LsSent)]
# print(Sent2Lemmas())
# print(Sent2Lemmas(nltk.word_tokenize(sQuote1)))
#         </pre>
#         </details>
# </font>
# <hr>

# ## Task 5
# 
# Complete the user-defined function (UDF), `Flatten()`, which takes a list of lists `LL` and returns a flattened list of all elements. The output of the function with default argument, `LL=[[1,2,3],[4, 3]]`, should be:
# 
#         >>> Flatten(LL=[[1,2,3],[4, 3]])
#         [1, 2, 3, 4, 3]
#         
# <b>Hint:</b> You can either use a double (i.e. nested) for-loop or <a href="https://www.python.org/dev/peps/pep-0202/#examples">nested list comprehension</a>.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# def Flatten(LL=[[1,2,3],[4, 3]]):
#     'LL: a list of lists'
#     return [w for L in LL for w in L]
# Flatten()
#         </pre>
#         </details>
# </font>
# <hr>

# ## Task 6
# 
# Create `LLsQuote2`, which should be a list of sentences from `sQuote2` that have each been made into a list of word token strings.
# 
# <b>Hint:</b> Your resulting list of lists should be in the format of <code>LLsWords</code> variable returned by <code>nltk.corpus.gutenberg.words('carroll-alice.txt')</code> above.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# LLsQuote2 = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(sQuote2)]
# print(LLsQuote2)
#         </pre>
#         </details>
# </font>
# <hr>

# ## Task 7
# 
# Complete a UDF called `Sents2Lemmas()` that takes a list of sentences (where each sentence is a list of word tokens) and lemmatizes the words in each sentence. The output should be returned as a list of lists or a list of all (lemmatized) words in the original sequence, if the `flatten` argument is true. Test your UDF on default arguments and on `LLsQuote2`. For example:
# 
# 
#         >>> Sents2Lemmas(flatten=True)
#         ['I', 'be', 'love', 'NLP', '.', 'NLP', 'be', 'fun', '.']
#         >>> Sents2Lemmas(flatten=False)
#         [['I', 'be', 'love', 'NLP', '.'], ['NLP', 'be', 'fun', '.']]
#         
# <b>Hint:</b> This is an individual application of <code>Sent2Lemmas()</code> to each sentence (list of words) stored in <code>LLsSents</code>. Use <code>Flatten()</code> to flatten the list of lists.   

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# def Sents2Lemmas(LLsSents=[['I', 'am', 'loving', 'NLP', '.'],['NLP','is','fun', '.']], flatten=True):
#     '''LLsSents: a list of sentences, which are in the form of lists of words
#     flatten: whether to flatten a list of lists before outputting it
#     Return: list of lemmatized words, if flatten is true. Otherwise, list of lists of lemmatized words'''
#     LLsWords = [Sent2Lemmas(sSent) for sSent in LLsSents]
#     return Flatten(LLsWords) if flatten else LLsWords
# Sents2Lemmas(flatten=True)
# Sents2Lemmas(flatten=False)
# Sents2Lemmas(flatten=False)
# print(Sents2Lemmas(LLsQuote2, flatten=True))
# print(Sents2Lemmas(LLsQuote2, flatten=False))
#         </pre>
#         </details>
# </font>
# <hr>

# # Task 8
# 
# 1. Create `LsLemmas`, which is a (flattened) list of lemmas from `LLsWords`.
# 1. Create a dictionary `DsOrigDist` of words as keys and their frequencies as values for the `LsWords` list of words. 
# 1. Create a dictionary `DsLemmaDist` of words as keys and their frequencies as values for the `LsLemmas` list of words. 
# 1. Print counts of the following words from the original and lemmatized texts: `'be'`, `'is'`, `'was'`, `'have'`, and `'had'`. 
# 
# Notice the drastic change in frequencies of word tokens after lemmatizations. For example, `'be'` occured 145 times in original text, but 796 times in lemmatized text. For example:
# 
#         >>> print('be:', DsOrigDist['be'], ' -> ', DsLemmaDist['be'])
#         be: 145  ->  796
#         
# <b>Hint:</b> Recall that <a href="https://docs.python.org/3/library/collections.html#collections.Counter"><code>collections.Counter()</code></a> returns a dictionary.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# LsLemmas = Sents2Lemmas(LLsWords, flatten=True)
# DsOrigDist = collections.Counter(LsWords)
# DsLemmaDist = collections.Counter(LsLemmas)
# 
# print('be:', DsOrigDist['be'], ' -> ', DsLemmaDist['be'])
# print('is:', DsOrigDist['is'], ' -> ', DsLemmaDist['is'])
# print('was:', DsOrigDist['was'], ' -> ', DsLemmaDist['was'])
# print('have:', DsOrigDist['have'], ' -> ', DsLemmaDist['have'])
# print('had:', DsOrigDist['had'], ' -> ', DsLemmaDist['had'])
#         </pre>
#         </details>
# </font>
# <hr>

# ## Task 9
# 
# Use results of previous task to find a word with the largest **increase** of its count due to lemmatization. This would be useful, for example, if you wanted to find the base word that had the most variants in the text.
# 
# Next, find the word with the largest **decrease** of its count due to lemmatization. This would be useful if you wanted to find the word that was most frequently replaced with a base word during the process.
# 
#   <b>Hint:</b> You can create a for-loop approach to search all elements of elements in each dictionary, but it might be easier to convert each dictionary into a Pandas DataFrame, set words as indices, and then <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html">join</a> on their indices. A dataframe allows you to easily take differences of the two frequency columns, sort by the difference column and take top 1 and bottom 1 rows.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#B31B1B>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# # Solution 1: 
# dfO = pd.DataFrame(DsOrigDist.items(), columns=['word','freqO']).set_index('word')
# dfL = pd.DataFrame(DsLemmaDist.items(), columns=['word','freqL']).set_index('word')
# dfOL = dfO.join(dfL)    # join on matching words
# dfOL['Diff'] = (dfOL.freqL - dfOL.freqO).fillna(0) # find increase in counts of lemmas
# dfDiff = dfOL.Diff.sort_values(ascending=False)
# dfDiff.head(1)  # top increased lemma
# dfDiff.tail(1)  # top decreased lemma
#  
# #Solution 2: 
# merged = {k: DsLemmaDist.get(k, 0) - DsOrigDist.get(k, 0) for k in set(DsOrigDist) & set(DsLemmaDist)}
# max(merged, key=merged.get)
# min(merged, key=merged.get)
#         </pre>
#         </details>
# </font>
# <hr>

# ## Task 10
# 
# Plot a bar chart with the top 50 words that had the greatest count **increase** due to lemmatization.
# 
# <b>Hint:</b> You can take top 50 rows in a dataframe and use <code>.plot.bar()</code> method of the resulting dataframe.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# dfDiff.head(50).plot.bar(figsize=(20,3), title='Increase in word counts after lemmatization', grid=1)
#         </pre>
#         </details>
# </font>
# <hr>

# ## Task 11
# 
# Plot a bar chart with the top 50 words that had greatest count **decrease** due to lemmatization.
# 
# <b>Hint:</b> you can take bottom 50 rows in a dataframe and use <code>.plot.bar()</code> method of the resulting dataframe.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=carnelian>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# dfDiff.tail(50).plot.bar(figsize=(20,3), title='Decrease in word counts after lemmatization', grid=1)
#         </pre>
#         </details>
# </font>
# <hr>

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video. 

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = 'all'    # allows multiple outputs from a cell
import re # import the regex library, re


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# A search with a string's [`replace()`](https://docs.python.org/3/library/stdtypes.html#str.replace) method can be very limited and slow. Using methods for finding and altering patterns with methods from the regex `re` library can make your searches much more efficient and save a lot of time. Review the code Professor Melnikov used to find and replace patterns in strings with a string's built-in replace method and with the `re` library's [`sub()`](https://docs.python.org/3/library/re.html#re.sub) method for working with strings.
# 
# ## Use a String's `replace()` Method
# 
# In the following example that uses string's `replace()` method, we replace the digits in a social security number (SSN) with asterisks. In each pass through the string a single type of digit is replaced, so replacing a full SSN requires ten passes through the string.

# In[2]:


sDoc = 'Social security number: 123-45-6789'
sDoc.replace('0', '*').replace('1', '*').replace('2','*').replace('3','*').replace('4','*')  .replace('5','*').replace('6','*').replace('7','*').replace('8','*').replace('9','*')


# <details style="margin-top:0px;border-radius:20px"><summary>
#     <div id="button" style="background-color:#eee;padding:10px;border:1px solid black;border-radius:20px">
#        <font color=#B31B1B>▶ </font> 
#         <b>Application in Web Security</b>: Anonymization of Personal Information
#     </div></summary>
# <div id="button_info" style="padding:10px">In the anonymization example above, ten passes over the given string were needed to replace all the digits with asterisks, <code>'*'</code>. If a corpus contains billions of characters or if the search pattern is very complex, this approach is computationally inefficient, costly, and time consuming. Performing a single pass search and replacing all of the digits at once saves valuable time and resources.
#     
# Furthermore, anonymizing only personally identifiable numbers, such as social security numbers (SSN), phone numbers, medical record numbers, internet protocol addresses and such, requires a search for a trigger token, say "SSN", and then a local search for the sensitive information in the vicinity of the trigger. This logic might be too complex for the <code>replace()</code> method.</div> </details>
# 
# ## Use the Regex `sub` method 
#     
# Regex search methods are often faster and cleaner than built-in string methods because regex can search over a string in one pass. Here, we use the [`sub()`](https://docs.python.org/3/library/re.html#re.sub) (substitution) method from the `re` library to perform the same anonymization task on the social security number. In this case, we search over a string of characters for the matching pattern of any digit, described as `'\d'` in our search.

# In[3]:


print(re.sub('\d', '*', sDoc))      # substitute a pattern with a replacement in a string
print(re.sub('[0-9]', '*', sDoc))   # equivalent search for a digit using a character class 0 through 9


# ## Additional Regex Features
# 
# Click the following button to reveal a short list of [basic pattern matching rules](https://www.regular-expressions.info/tutorial.html). You will use some of these rules in the tasks below to gain experience using regex.
# <div id="blank_space" style="padding-top:25px"><details><summary><div id="button" style="color:white;background-color:#de2424;padding:10px;border:3px solid #B31B1B;border-radius:30px;width:235px;padding-left:25px;float:left;margin-top:-20px"> 
#     <b>Pattern Matching Rules →</b>
#     </div></summary>
# <div id="button_info" style="padding:20px;background-color:#eee;border:3px solid #aaa;border-radius:30px;margin-left:25px"><p style="padding:15px 2px 2px 2px">
#    Here is a short list of the basic pattern matching rules. Try these out to gain expertise and experience these in action.
# 
# |Rule|What It Searches For|
# |:---:|:---|
# |`.`|Any single character. If a period is placed inside a character class, such as `[.]`, then it matches a period only.|
# |`^`|The start of the string.|
# |`$`|The end of the string.|
# |`\b`|A word boundary. For example, `r'\bthing\b'` matches `'thing'` word (surrounded by spaces or punctuation), but not `'nothing'` or any other word with a subword `'thing'`. Recall that `r'...'` is a raw string.|
# |`?`|Zero or one of the previous pattern.|
# |`*`|Any number of repeated cases of the previous pattern.|
# |`+`|One or more numbers of repeated cases of the previous pattern.|
# |`[]`|A [character class](https://www.regular-expressions.info/charclass.html). For example, `[0-9a.]`matches any digit, letter "a" or a period.|
# |`[^]`|Any character excluded from the square brackets containing symbols after `^`.|
# |<code>\|</code>|Any pattern on the left or the right of the pipe symbol.|
# |`\d` or `[0-9]`|Any decimal digit. The dash represents the range of digits.|
# |`\D` or `[^0-9]`|Any non-decimal digit.|
# |`\s`|Whitespace characters, including `' '`, `'\t'`, `'\n'`, `'\r'`.|
# |`\S`|Non-whitespace characters.|
# |`\w` or `[a-zA-Z0-9]`|Any alphanumeric character.|
# |`\W` or `[^a-zA-Z0-9]`|Any non-alphanumeric character.|
# |`()`|A [match group](https://regexone.com/lesson/capturing_groups). For example, `(he|we|they)` matches any of the listed pronouns in a string.|
# |`{m}`|The preceding element must repeat exactly `m` times. For example, `'i{4}'` is the same as `'iiii'`.|
# |`{m,n}`|The preceding element must repeat between `m` and `n` times.|
# |`{m,}`|The preceding element must repeat at least `m` times.|
# 
# </p></div> </details></div>
# 
# <p>&nbsp;</p>
# Additionally, to ensure that your function is interpreted properly, you may need to disable the effect of a special regex pattern within a function. You can do this escaping by prefixing patterns with a backslash <code>\</code>. Click the following button to reveal some important ways to use escaping. 
# 
# <div id="blank_space" style="padding-top:25px"><details><summary><div id="button" style="color:white;background-color:#de2424;padding:10px;border:3px solid #B31B1B;border-radius:30px;width:235px;padding-left:25px;float:left;margin-top:-20px"> 
#     <b>Escaping Special Patterns →</b>
#     </div></summary>
# <div id="button_info" style="padding:20px;background-color:#eee;border:3px solid #aaa;border-radius:30px;margin-left:25px"><p style="padding:15px 2px 2px 2px">
# 
# |Rule|What It Searches For|
# |:---:|:---|
# |`\.`|Escapes period's super powers and makes it match a period only.|
# |`\[`|Escapes the start of the character class brackets and simply matches a square bracket. For example, <code>[ab]</code> matches <code>a</code> or <code>b</code>, but <code>\[ab\]</code> literally matches <code>[ab]</code>.|
# |`\?`|Literally matches a question mark (not any single character). For example, <code>H?</code> matches <code>Hi</code>, <code>H1</code>, <code>H.</code>, and any other character following letter <code>H</code>, but <code>H\?</code> literally matches <code>H?</code> only.|
# |`\+`|Literally matches a plus sign (not a preceding&nbsp;character).|
# 
# </p></div> </details></div></div></div>

# You can explore regex more in [regex101](https://regex101.com/r/NwlUlO/1), an interactive online regex tool which visually explains regex processing on a test string.

# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # Optional Practice
# 
# Here, you'll practice some basic regex patterns that were introduced in the previous video. Each of these tasks requires you to use the regex method `sub`. You may also need to consult the list of pattern matching rules, above, as you work through these tasks. Note that many of these problems can be solved in several ways.
# 
# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.
# 
# Run the following code to load and print the string `sQuote`. `sQuote` contains a mix of cased characters, adjacent whitespace characters, periods, multiple sentences, etc. You will work with `sQuote` over the next few exercises as you practice using regex patterns. 

# In[4]:


sQuote = """`If you live long enough, you'll make mistakes.  
But if you learn from them, you'll be a better person.  
It's how you handle adversity, not how it affects you.  
The main thing is never quit, never quit, never quit.`
~ A quote by William J. Clinton, 42nd U.S. President from 1993 to 2001
"""
print(sQuote)  # formatted
sQuote         # unformatted


# ## Task 1
# 
# Use `re` library to replace `'you'` with `'we'` in `sQuote`.
# 
# <b>Hint:</b> You can search for a string <code>'you'</code> and replace with another string using the <code>.sub()</code> method of <code>re</code> object.

# In[5]:


re.sub('you', 'we', sQuote)  # replace 'you' with 'we'


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>See <b>solution</b>.</summary>
#             <pre>
# re.sub('you', 'we', sQuote)  # replace 'you' with 'we'
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 2
# 
# Use `re.sub()` to replace two or more adjacent spaces with a single space character.
# 
# <b>Hint:</b> Use <code>'+'</code> to find multiple adjacent versions of a single string.
# 

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>See <b>solution</b>.</summary>
#             <pre>
# re.sub(' +', ' ', sQuote)  # replace two or more adjacent spaces with a single space character
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 3
# 
# Use a technique that uses escaped characters with `re.sub()` to replace two or more adjacent whitespaces with a single space character.
# 
# <b>Hint:</b> <code>'\s'</code> represents a whitespace character class.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>See <b>solution</b>.</summary>
#             <pre>
# re.sub('\s+', ' ', sQuote)  # replace two or more adjacent whitespaces with a single space character
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 4
# 
# Use `re.sub()` to expand the contraction `"'ll"` to the word `' will'`.
# 
# <b>Hint:</b> Wrap single quotes inside double quotes, <code>"'ll"</code>, or use escaping with the single quote as <code>'\'ll'</code>.
# 

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>See <b>solution</b>.</summary>
#         <details>
#             <pre>
# re.sub("'ll", ' will', sQuote)  # expand contraction "'ll" with ' will'
#             </pre>
#         </details>
#     </details> 
# </font>
# <hr>

# ## Task 5
# 
# Use `re.sub()` to replace a 4 digit year with a word `'YEAR'`.
# 
# <b>Hint: </b> Use sequential digit searches.
# 

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>See <b>solution</b>.
#         </summary>
#             <pre>
# re.sub('\d\d\d\d', 'YEAR', sQuote)  # solution 1: replace a 4 digit year with a word 'YEAR'
# re.sub("\d{4}", 'YEAR', sQuote)     # solution 2: replace a 4 digit year with a word 'YEAR'
# </pre>
#         </details>
# </font>
# <hr>

# ## Task 6
# 
# Use `re.sub()` to replace `'U.S.'` with `'U.S.A.'` .
# 
# <b>Hint:</b> Escape a period character to avoid it being interpreted as any character.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>See <b>solution</b>.
#         </summary>
#             <pre>
# re.sub('U\.S\.', 'U.S.A.', sQuote)  # replace 'U.S.' with 'U.S.A.'. We escape a period in re pattern, not in replacement
#             </pre>
#     </details>
# </font>
# <hr>

# ## Task 7
# 
# Use `re.sub()` to replace `'William J.'` with `'Bill'`.
# 
#  <b>Hint:</b> Search for the full string <code>'William J.'</code> .

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>See <b>solution</b>.
#         </summary>
#             <pre>
# re.sub('William J\.', 'Bill', sQuote)  # replace 'William J.' with 'Bill'
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 8
# 
# Use `re.sub()` to replace sequential instances of `'never quit'` with a single instance of `'never quit'`.
# 
# <b>Hint:</b> You can search for multiple version of any text (not just individual characters) by placing it inside parentheses. 

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>See <b>solution</b>.
#         </summary>
#             <pre>
# re.sub('(never quit, )+', '', sQuote)  # replace multiple 'never quit' with a single 'never quit'
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 9
# 
# Use `re.sub()` to replace all word characters and punctuation characters with `'*'`.
# 
# <b>Hint:</b> Try using a character class <code>[]</code> with all characters you want to find. Recall: `'\w'` is a word character, and can be a letter, digit, an underscore, or a tilde.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>See <b>solution</b>.
#         </summary>
#             <pre>
# re.sub('[\w.,~`\']', '*', sQuote)  # solution 1: replace all word and punctuation characters with '*', or, alternatively
# re.sub('\S', '*', sQuote)          # solution 2: replace all non-whitespace characters with '*'
# re.sub('[\w.,~`\']', '*', sQuote) == re.sub('\S', '*', sQuote)  # compare resulting strings
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 10
# 
# Use `re.sub()` to replace all non-word characters with `'_'`.
# 
# <b>Hint:</b> Consider <code>'\W'</code>, which is any non-word character, i.e., not a letter, not a digit, not an underscore.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>See <b>solution</b>.
#         </summary>
#             <pre>
# re.sub('\W', '_', sQuote)  # replace all non-word characters with '_'
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 11
# 
# Use `re.sub()` to replace `'the'` and `'The'` with `'***'`.
# 
# <b>Note:</b> Words containing the strings, `The` or `the`, will be partially replaced.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>See <b>solution</b>.
#         </summary>
#             <pre>
# re.sub('[Tt]he', '***', sQuote)  # solution 1: replace 'the' and 'The' with '***'
# re.sub('(the|The)', '***', sQuote)  # solution 2: replace 'the' and 'The' with '***'
#             </pre>
#         </details>
# </font>
# <hr>

# ## Task 12
# 
# Use `re.sub()` to replace `'you'` or `'it'` or `'thing'` with `'****'`.
# 
# <b>Hint:</b> You could use a match group.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>See <b>solution</b>.</summary>
#             <pre>
# re.sub('(you|it|thing)', '****', sQuote)  # replace 'you' or 'it' or 'thing' with '****'
#             </pre>
#     </details> 
# </font>
# <hr>

# ## Task 13
# 
# Use `re.sub()` to replace the **word** `'be'` with `'*****'`. Thus, a word `'better'` should remain unchanged.
# 
# <b>Hint:</b> You could use a word boundary, <code>'\b'</code>, to search the exact word and not a subword. You may also need to use <code>r'string'</code> to indicate a raw string. Otherwise, you can use double slashes as escapes.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details>
#         <summary><font color=#b31b1b>▶</font>See <b>solution</b>.
#         </summary>
#             <pre>
# re.sub(r'\bbe\b', '*****', sQuote)  # replace a word 'be' with '*****'
# re.sub('\\bbe\\b', '*****', sQuote)  # replace a word 'be' with '*****'
#             </pre>
#         </details>
# </font>
# <hr>

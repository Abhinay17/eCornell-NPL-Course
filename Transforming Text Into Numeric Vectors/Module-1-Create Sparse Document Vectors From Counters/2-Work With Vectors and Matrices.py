#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import numpy as np, matplotlib.pyplot as plt


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review**
# 
# In this notebook, you'll continue building your foundation of linear algebra by examining matrices, which can be thought of as vectors of vectors (or a collection of identically sized vectors stacked together). Matrices are typically structures with multiple rows and multiple columns. 
# 
# The rows in a matrix are sometimes referred to as "observations" while the columns are sometimes referred to as "features," "variables," or "dimensions." We specify the shape of a matrix by the number of rows x number of columns. For example, a matrix with three rows and five columns would be referred to as a 3x5 matrix. This matrix would also be referred to as being a data set with five dimensions.
# 
# The term "dimensions" can be confusing, however, because it has two different ways it can be used. If you are talking about the number of columns in a matrix or the number of items in a vector, these numbers are sometimes referred to as dimensions, such as the five-dimensional matrix example from the previous paragraph. Yet sometimes you will see vectors referred to as 1D (or one-dimensional) objects and matrices as 2D (or two-dimensional) objects regardless of the number of elements they contain. In this context, the term "dimension" is referring to the shape of the object, not the number of features or elements in the object. For example, the 3x5 matrix above could be said to have two dimensions in this context (rows and columns), whereas a vector could be said to have only one dimension (a single array of values, without specificity as to whether it is row or column oriented).
# 
# Usually any confusion can be cleared up by the context in which the term "dimension" is used as long as you are aware that the meaning may change based on context. If you are ever unclear about the meaning of the term as used in this or any other materials in the course, please feel free to reach out to your facilitator for guidance.

# ## Matrices
# 
# Matrices can have any number of rows and columns. First, use the NumPy array function to define a $2\times2$ matrix.

# In[2]:


A = np.array(
    [[1,-2],
     [3,4]])
A              #print the matrix
A.shape        #print the matrix shape


# Once you've defined a matrix, you can use Python to perform operations on matrices by following the rules of matrix algebra. For example, **scalar multiplication** is an element-wise operation, where each element of a matrix is multiplied by the scalar value.

# In[3]:


10 * A   # scalar multiplication 


# **Matrix transposition** is an operation that "flips" the matrix along its diagonal line (from top left to bottom right). The rows become columns and the columns become rows. Double transposition returns the matrix to original form. In fact, any pair of transpose operations cancel each other out. 

# In[4]:


B = A.T          # transposition
print("B = \n {}".format(B)) # Prints b, with a "\n" newline character to make the matrix more readable
print(' ')
print("B.T = \n {}".format(B.T))     # back to the original matrix A      
A.T.T.T.T == A   # check if original matrix is returned, ""=="" is also element-wise
A.T.T.T.T == B   # returns true for elements which stayed the same, and false for values that changed


# You can also add and subtract matrices, but these are element-wise operations, so to do so the matrices must have the same dimensions (i.e., number of rows and columns). Trying to add or subtract matrices with different shapes will result in an error.
#     
# Adding a **square matrix** (a matrix in which the number of rows is equal to the number of columns) to its own transpose results in a [**symmetric matrix**](https://mathworld.wolfram.com/SymmetricMatrix.html), which has the same values on the opposite sides of its diagonal. 

# In[5]:


S = A + B   # add matrix to its transpose
print("The sum of A + B = \n {}".format(S))           # a symmetric matrix
print("Note that the values in the top right and bottom left (opposite of the diagonal values) are the same.")
print(" ")
S == S.T    # Another way to check if S is symmetric - the top right and bottom left values should be True


# As we have seen, you can check the shape of a NumPy array with the [`shape`](https://numpy.org/doc/stable/reference/generated/numpy.shape.html) attribute.

# In[6]:


A.shape  # returns a tuple with a number of rows and columns
B.shape


# You can stack matrices together using several methods, including NumPy's `hstack` and `vstack` functions, as long as the shape along the axis being added to matches. For square matrices, this does not matter since both axes are the same value, but for non-square matrices, attempting to stack along an axis with non-matching values will cause an error. See the examples below and make sure you understand which shape values matter depending on the operation you're trying to perform.

# In[7]:


print("Using hstack with A and B:")
H1 = np.hstack([A,B])
print(H1)

print("H1 shape = {}".format(H1.shape))
print(" ")

print("Using vstack with A and B")
V1 = np.vstack([A,B])
print(V1)

print("V1 shape = {}".format(V1.shape))
print(" ")

print("Stack different-shaped matrices horizontally:")
print("H1 and A will work because they have the same number of rows")
H2 = np.hstack([H1,A])
print(H2)

print("H2 shape = {}".format(H2.shape))
print(" ")


print("H1 and V1 will fail because the value for rows is not the same")
#H3 = np.hstack([H1,V1])  #Uncomment this code to see this operation fail
print(" ")

print("Stack different-shaped matrices vertically:")
print("V1 and A will work because they have the same number of columns")
V2 = np.vstack([V1,A])
print(V2)

print("V2 shape = {}".format(V2.shape))
print(" ")

print("H1 and V1 will once again fail, this time because the number of columns does not match")
#V3 = np.vstack([V1,H1]) #Uncomment this code to see this operation fail


# ## Sparse Matrices
# 
# <span style="color:black"> Highly sparse matrices are common in NLP. These matrices contain millions of entries, of which only ~1% (or even less) are non-zero entries. These matrices are ubiquitous in NLP because when you use a matrix to track counts of specific words in sentences, the rows represent sentences or documents and columns represent the terms or words in those sentences. Each sentence uses only a few words from the total vocabulary (which can be very large), so most word count values are zeros. 
#     
# <span style="color:black">Storing large matrices takes up computer memory, so you should avoid storing highly sparse matrices in their raw form. One efficient way to store these matrices is by storing indices of non-zero locations and their values using `scipy.sparse`. Explicit storage of indices is an overhead, but you are still saving a significant storage space, if 99% or more zeros are not stored.
#     
# Note: There is no uniform definition for a "sparse matrix". Typically, we describe a matrix with "lots of zero values" as sparse and can say that one matrix is more or less sparse than another. However, one can define sparsity based on values "close" to zero or even close to zero only in some "important" positions in the given matrix.

# In[11]:


from scipy.sparse import csr_matrix

X = csr_matrix([[0, 1, 0], [0, 0, 3], [4, 0, 0]])  # sparse matrix data structure
print("X object (all rows have one value, but in different positions):")
X
print(' ')

print("First element of X:")
X[0]
print(' ')

print("Second element of X:")
X[1]
print(' ')
print("Third element of X:")
X[2]

#X[3] #There are only 3 rows, so there is no index position 3

print(' ')
print("********")
print(' ')

X2 = csr_matrix([[0, 1, 1], [0, 0, 0], [4, 0, 0]])  # sparse matrix data structure

print("X2 object (first row two values, second row all 0's, third row one value)")
X2
print(' ')

print("First element of X2:")
X2[0]
print(' ')

print("Second element of X2:")
X2[1]
print(' ')

print("Third element of X2")
X2[2]
print (' ')

print("Note that both X and X2 report having 3 stored elements, but each sub-element reports the number of non-zero values in a given row")


# You can convert SciPy sparse matrices to NumPy matrices with the `.toarray` method.

# In[12]:


print("Convert X to a NumPy matrix")
Y = X.toarray()  # convert to numpy (dense) array
Y
print(' ')

print("Convert X2 to a NumPy matrix")
Y2 = X2.toarray()
Y2


# You can slice sparse matrices in the same manner in which you can slice NumPy matrices. Sparse matrices remain sparse after they are sliced. Below, retrieve a top left $2\times 2$ submatrix from the SciPy sparse matrix $X$ and the NumPy matrix $Y$.

# In[13]:


X[:2,:2]
Y[:2,:2]


# ## Loops (and List Comprehensions) vs. Vectorization

# Occasionally, you may need to iterate over the rows of a matrix. A common approach to performing this is to use a loop or list comprehension.

# In[14]:


for y in Y : print(y)   # regular loop over rows of Y


# To iterate over columns, you might transpose the matrix then apply the for loop.
# 
# However, it is also useful when working with large data to avoid loops in Python by using vectorization approaches and, where appropriate, in place of loops. For example, say you want to only print out rows where the first value of Y equals 0. You could write a loop like this:

# In[15]:


for y in Y: 
    if y[0] == 0:
        print(y)


# However, a more efficient way to write this would be to use vectorization. Do this by specifying the condition that should be true as a value for subsetting the operation to be performed. For example, the code below in a single line (and single Python operation) produces the same output as the loop above:

# In[16]:


print(Y[Y[:,0]==0])


# To evaluate how this is working, you can print just the condition that is being evaluated. This returns an index of True / False values, and only those index positions where the condition is True will perform the operation (in this case, the "print" statement):

# In[17]:


print(Y[:,0]==0)


# The vectorized approach takes only one Python operation performed over the entire data set at once, whereas the loop approach takes three - one operation for each row. While this may seem insignificant, consider the case where Y was three billion rows instead of just three. The loop approach would initiate three billion separate Python operations and take a considerable amount of time. The vectorized approach, while it would still take some time, would likely be anywhere from **10x to 30x** faster (or better, even) since only two Python operations are being executed, albeit over much larger chunks of data.
# 
# While list comprehensions have some performance gains compared to traditional loops, vectorized code will still outperform them many times over. On large data, the perfomance gains are substantial.
# 
# Deep dives into vectorization as loop alternatives in code go beyond the scope of this class. However, in a production environment with significantly large data, they can take a process than might run for several hours, days, or more and complete them in only hours or minutes. 

# ## Dot Product
# 
# A **dot product** is a useful operation in machine learning because it provides a simple mathematical way of comparing similarity between vectors. Same-sign values result in positive contribution to the final sum, while the opposite-sign values yield a negative contribution. Thus, the higher the final dot product value, the more similar the vectors are interpreted to be. Conversely, the lower the dot product value, the more dissimilar the two vectors can be interpreted to be.
# 
# For example, the two vectors below have zero dot product because of equal opposite-sign contribution to the sum (1x1 = 1, 1x-1 = -1, and the sum of 1 and -1 = 0). Note the directions of these vectors (try drawing these on a piece of paper using the two values in each as x and y coordinates). Also, since the dot product is zero, we know that these are two orthogonal (i.e., perpendicular) vectors.

# In[18]:


np.array([1, -1]) @ np.array([1,1])


# However, the following two vectors have a higher dot product because of all positive contributions to the sum. Note the directions of these vectors

# In[19]:


np.array([1, -1]) @ np.array([1, -1])


# Thus, we could use this as a metric to conclude that the second set of vectors is more similar than the first set.

# If we consider the matrix `Y` as a box of row vectors, you can compute dot product of each row with another same-size vector using a loop as show next.

# In[20]:


a = np.array([1,1,1])      # create a 1D array of ones
for y in Y: print(y @ a)   # iterative computation of dot products of rows with a vector


# However, similar to the example above where we printed rows based on a condition, this method is overly complex and inefficient. It is again slow because the loop itself is done at Python level, one row at a time. In this case, the NumPy dot product libraries allow us to use the `@` operator to perform this operation on the entire matrix at once. The first thing NumPy functions do is convert the Python operation to a lower-level compiled language (like C), which comes with substantial performance gains - even more than vectorization on large data sets. 
# 
# The dot product of a matrix $Y$ below by a vector $a$ can be mathematically written as $Ya$, skipping the multiplication symbol for brevity.

# In[21]:


Y @ a


# For matrices, this dot product operation is not commutative and you will receive a different answer if the `a` is on the left of the matrix `Y`. In this situation, the vector is multiplied by every column, instead of every row.
# 
# The right multiplication of a vector $a$ below by a matrix $Y$ is mathematically written as $aY$.

# In[22]:


a @ Y


# To avoid mistakes, make sure you understand the order of operations between matrices and vectors. 
# 
# ## Deeper Dive: Dot Product of Two Matrices
# 
# The dot product of two matrices $A\cdot B$ or $AB$ is basically the dot products of rows from the left matrix $A$ and columns from the right matrix $B$. The result is a matrix, which we will call $C$, with the number of rows matching the number of rows in $A$ and the number of columns matching the number of columns in $B$. Each $ij$-th value of $C$ is a dot product of row $i$ in $A$ and column $j$ in $B$. 
# 
# This is usually not commutative, so $BA$ is likely to produce a different result (assuming it will produce a result at all). Indeed, dot products only work if the number of rows in $A$ is the same as the number of columns in $B$. Put another way, the "inner" values of the matrices must match (dot product is also known as "inner" product), but the "outer" values can be different. For example, you can compute the dot product of a 3x6 matrix with a 6x9 matrix, which would result in a 3x9 matrix. However, you could not perform a dot product calculation of a 3x6 with another 3x6 matrix, since the inner values (6 and 3, respectively) are not the same.
# 
# Note the difference between dot products, which uses the `@` symbol, and element-wise multiplication, which uses the `*` symbol. (Element-wise multiplication is also referred to as "Hadamard product.") For element-wise operations, the shape of the matrices must be identical. Therefore, while a 3x6 `@` 3x6 operation would fail, a 3x6 `*` 3x6 would succeed, and the resulting matrix would also be a 3x6.   
# 
# In NumPy, you can perform 2D (row x column) matrix multiplication with the `@` symbol, [`np.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html), [`np.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) or [`np.ndarray.dot`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dot.html). (Note that when working with multidimensional arrays, there are some cases where `@`, `np.matmul`, and `np.ndarray.dot` may produce different results; however, when using 2D [row x column] arrays in this course, you can use them interchangeably.)
# 
# See examples below:

# In[23]:


Y.shape # Since this is a square matrix, the inner values will match for dot product calculation
Y @ Y   
np.dot(Y, Y)
Y.dot(Y)
np.matmul(Y, Y)
print("Note that all functions return the same results")


# Create two non-square matrices with shapes `(2,3)`  and `(3,2)`, which can be represented mathematically as $2\times3$ and $3\times 2$.

# In[24]:


Y23 = Y[:2,:]  # slice a 2 by 3 submatrix
Y32 = Y[:,:2]  # slice a 3 by 2 submatrix
Y23
Y32


# Now you will verify the dimensionality match requirement by examining the dimensions of matrices that result from matrix multiplication. 

# In[25]:


print("2x3 @ 3x2 dot product")
Y23 @ Y32    # inner shape values are both 3, and the result is a 2x2 matrix (the outer shape values)
print(' ')

print("3x2 @ 3x2 dot product")
Y32 @ Y23    # inner shape values are both 2, and the result is a 3x3 matrix (the outer shape values)
print(' ')

print("Try element-wise multiplication of 3x2 and 2x3 matrices")
print(' ')
try:
    Y32 * Y23    # throws an error since Hadamard product requires matching shapes
except ValueError as e:  # catch an error into an object e
    print(e, '\n')             # print error attributes    
print(' ')

print('********')
print(' ')

print("Try dot product of 2x3 and 2x3 matrix")
print(' ')
try:
    Y23 @ Y23    # throws an error about mismatched inner dimensions in matmul operation
except ValueError as e:  # catch an error into an object e
    print(e)             # print error attributes       


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Optional Practice**
# 
# You will now practice creating matrices and performing matrix operations.
# 
# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.

# ## Task 1
# 
# Create a $3\times 3$ matrix $W$ (using the NumPy array function) with the values:
# 
#     [[1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9]]
# 
# Then, compute $W + W,\,\,\,\,\,10W - 5Y,\,\,\,\,\,\,WW / 10$
# 
# **Hint:** Check code above, where we used the same operations. 

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
# <pre>
# W = np.array([[1,2,3],[4,5,6],[7,8,9]])
# W
# W + W
# 10 * W - 5 * Y
# W @ W / 10
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# Compute $(W + W^T)(W + W^T)$, where $W^T$ is the transpose of $W$ matrix.
# 
# <b>Hint:</b> See code above where we used the same operations.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
# <pre>
# (W + W.T) @ (W + W.T)
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# Define a $3\times 3$ identity matrix with [`np.eye()`](https://numpy.org/devdocs/reference/generated/numpy.eye.html) method and save it to a variable $I$. This is a matrix of ones on diagonal and zeros off diagonal. Then compute on paper: 
# 
# $$IW,\,\,\,WI,\,\,\,I\odot W,\,\,\,W\odot I$$
# 
# <b>Hint:</b> Follow the link to view the documentation on <code>np.eye()</code> method. Also recall that $\odot$ is a Hadamard (element-wise) product.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# I = np.eye(3)   # identity matrix with ones on diagonal and zero elsewhere
# I @ W           # left or right dot product with I retains W without a change
# W @ I
# I * W           # element-wise multiplication with I wipes out non-diagonal elements
# W * I
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 4
# 
# Define $F$ as the top left $2\times 2$ submatrix of $W$ and $G$ as the top right $2\times 2$ of $W$. Then compute on paper: 
# 
# $$F \odot G, \,\,\,G \odot F,\,\,\, FG,\,\,\, GF$$
# 
#  <b>Hint:</b> See code above.

# In[ ]:


# check solution here


# 
# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# F, G = W[:2,:2], W[:2, 1:]
# F * G
# G * F
# F @ G
# G @ F
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 5
# 
# Define $F$ as the first two columns of $W$ and $G$ as the top two rows of $W$. Then compute on paper (and check with code): 
# 
# $$F \odot G, \,\,\,G \odot F,\,\,\, FG,\,\,\, GF,\,\,\, FF,\,\,\, F\odot F$$
# 
#  <b>Hint:</b> See code above.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# F, G = W[:,:2], W[:2,:]
# # F * G # throws an error since element-wise operations do not make sense on matrices with mismatched dimensions
# # G * F # also throws an error
# F @ G
# G @ F
# # F @ F   # also throws an error. Inner dimensions must match!
# F * F
#     </pre>
#     </details> 
# </font>
# <hr>

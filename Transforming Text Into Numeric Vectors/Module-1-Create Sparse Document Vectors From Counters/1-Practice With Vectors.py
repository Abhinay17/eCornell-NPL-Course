#!/usr/bin/env python
# coding: utf-8

# # **Setup**
#  
# Reset the Python environment to clear it of any previously loaded variables, functions, or libraries. Then, import the libraries needed to complete the code Professor Melnikov presented in the video.

# In[1]:


get_ipython().run_line_magic('reset', '-f')
from IPython.core.interactiveshell import InteractiveShell as IS
IS.ast_node_interactivity = "all"    # allows multiple outputs from a cell
import numpy as np, matplotlib.pyplot as plt  # load NumPy and visualization libraries


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Review** 
# 
# In this Jupyter Notebook, you will practice visualizing and performing operations on basic linear algebra structures. You'll use these techniques to calculate vector-encoded similarities among documents later in this module.

# <h3 style="color:black"> NumPy Arrays </h3>
# 
# <span style="color:black">The user-defined function (UDF) `Vec()` plots a 2-dimensional (2D) vector, i.e., array with two real-valued coordinates. This function is useful for visualizing 2D vectors and transformations.</span>

# In[2]:


def Vec(x=[1,2], text='', col='black', width=0.001):
    ''' Plots a vector x with 2 coordinates in color col with a specified text 
            and in the specified width.'''
    ax.arrow(x=0, y=0, dx=x[0], dy=x[1], width=width, head_width=0.2, head_length=0.1, fc=col, ec=col);
    ax.text(x[0], x[1], s=text, size=15, color='black');


# In Python, 2D vectors can be defined as 1D NumPy arrays. Create two arrays, `a` and `b`, that represent two 2D vectors.

# In[3]:


a = np.array([1,2])  # create a numpy array with two coordinates
b = np.array([2,1])

# Examine the arrays. Note the difference when calling the variables vs. using them in a print statement.

a
b

print("a = {} and b = {}".format(a,b))


# You can use the [`shape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html) attribute to retrieve the dimensions of a NumPy array structure. Thus, you can confirm that the vectors are of appropriate dimensionality before you apply vector operations on them. For example, we cannot add a 2D vector to a 3D vector. Occasionally, you will encounter a NumPy error and will need to investigate which of the vectors has an unexpected dimensionality and why.

# In[4]:


a.shape  # returns a tuple with number of values stored in each dimension of NumPy array


# Notice a single set of square brackets, `[...]`, in a definition of a vector. To define a vector of vectors (a.k.a. a *matrix* or table of values), you need to use a list of one or more equi-sized lists, `[[...],...,[...]]`, as you shall see below. 

# We begin by examining a few simple vectors created using the NumPy array function:

# In[5]:


vec1 = np.array([5])     # a 1D vector with a single value at index 0
vec2 = np.array([5,6])   # a 2D vector with a single value at indices 0 and 1
vec3 = np.array([5,6,7]) # a 3D vector with a single value at indices 0, 1, and 2

vec1
vec2
vec3

vec1.shape
vec2.shape
vec3.shape

vec3[1] # This will return the value in index position "1" for vec3


# Next, create the same objects but this time enclose the bracketed numbers inside a list and observe the differences in the outputs:

# In[6]:


mat1 = np.array([[5]])     # a 1x1 matrix with a single value at index 0,0
mat2 = np.array([[5,6]])   # a 1x2 matrix with a single value at indices 0,0 and 0,1
mat3 = np.array([[5,6,7]]) # a 1x3 matrix with a single value at indices 0,0, 0,1, and 0,2

mat1
mat2
mat3

mat1.shape  # Note that two axes are reported for the shape, even though there is only one number in the matrix
mat2.shape
mat3.shape

mat3[0,1] # Note that the matrix version must be given coordinates across both axes to produce the right value

#mat3[1] # This would fail because it is looking for row "1" which does not exist in this matrix.
         # Feel free to uncomment the line of code and see for yourself.


# Note the difference in the output of the vector converted to a matrix vs. the simple vector. Another way to perform this simple conversion would be to use the NumPy reshape function:

# In[7]:


vec1tomatrix = vec1.reshape([1,1])
vec1tomatrix
vec1tomatrix.shape


# Now create a matrix with more than one row by supplying a list of lists to the NumPy array function:

# In[8]:


mat2x1 = np.array([[5],[6]])             # a 2x1 matrix
mat2x2 = np.array([[5, 6],[7,8]])        # a 2x2 matrix
mat2x3 = np.array([[5, 6, 7],[4, 3, 2]])   # a 2x3 matrix
mat3x2 = np.array([[5,6],[7,4],[3,2]])   # a 3x2 matrix

mat2x1
mat2x2
mat2x3
mat3x2

mat2x1.shape
mat2x2.shape
mat2x3.shape
mat3x2.shape


# Note that each row in the matrix must have the same number of columns, otherwise what gets created is a list of lists, not a matrix. Check the output of the code below.

# In[9]:


badmat1 = np.array([[1, 2],[3, 4, 5]]) # Attempt to create an uneven matrix
badmat2 = np.array([[3, 4],[1, 2, 3]]) # Second attempt, same shape as the first
badmat1  
badmat2


# Note that if we try to add these two objects together, Python will concatenate the values in each list. This is not what we are trying to do with matrix addition, which will be demonstrated later.

# In[10]:


badmat1 + badmat2


# <h3 style="color:black"> Vector (and Matrix) Math Operations </h3>
# 
# You can perform the following operations on NumPy arrays and matrices:
# 
# * Addition, subtraction, multiplication, and division of two vectors (or matrices) is element-wise, where the operation is performed on corresponding coordinates across the two objects.
# * Operations with a scalar are element-wise, such the operation is performed on every coordinate with that scalar.
# 
# These operations are defined and used in *linear algebra*, which is a subfield of mathematics concerning the study properties of vectors, matrices, and higher-level structures called tensors.

# Take a look at some simple examples using the a and b vectors created earlier in the notebook:

# In[11]:


print("elementwise addition a + b")
print(a, '+', b, '=', a + b) # elementwise addition
print("********")
print("scalar addition a + 3")
print(a, '+', 3, '=', a + 3) # scalar addition
print("********")
print("elementwise subtraction a - b")
print(a, '-', b, '=', a - b) # elementwise subtraction
print("********")
print("scalar subtraction example a - 3")
print(a, '-', 3, '=', a - 3) # scalar subtraction
print("********")
print("elementwise subtraction b - a")
print(b, '-', a, '=', b - a) # elementwise subtraction
print("********")
print("scalar subtraction b - 5")
print(b, '-', 5, '=', b - 5) # scalar subtraction
print("********")
print("elementwise multiplication b * a")
print(b, '*', a, '=', b * a) # elementwise multiplication
print("********")
print("scalar multiplication a * 2")
print( '2 *', a, '=', 2 * a) # multiplication by a scalar
print("********")
print("scalar multiplication b * 2")
print( '2 *', b, '=', 2 * b) # multiplication by a scalar


# Note that each operation above results in a new vector in the same 2D (vector) space, not in a larger or smaller vector space. You can visualize this by plotting the two input vectors and the output vectors from the various algebraic manipulations in the same plot.

# In[12]:


ax = plt.axes()            # create an axis panel for plotting
Vec(a, 'a', width=0.1)     # plot vector a with text 'a' in black color with width 0.1
Vec(b, 'b', width=0.1)
Vec(a+b, 'a + b', 'green') # plot a vector resulting from the sum of a and b vectors
Vec(a-b, 'a - b', 'brown')
Vec(b-a, 'b - a', 'orange')
Vec(2*a, '2a', 'gray')
Vec(-0.5*a, '-0.5a', 'lightblue')

ax.set_title('Vector Algebra'); # create a title for the panel/figure
ax.set_xlabel('x');             # add a label for x axis
ax.set_ylabel('y');
plt.grid();                     # plot a mesh of vertical/horizontal gray lines
plt.xlim(-2,5);                 # define the range of visible horizontal interval
plt.ylim(-2,5);
plt.tight_layout();             # remove unnecessary white space in plot margins
plt.show();                     # print the plot on screen


# As demonstrated earlier, you can also store an equivalent of a 2D vector as a higher-dimensional NumPy structure, called a matrix. For example, the two vectors from above, `a` and `b`, can be stored as horizontal, one-row matrices `aRow` and `bRow` by adding an extra set of brackets around the numbers.
# 
# Matrices are a more compact and convenient way of combining many (for example, millions, hundreds of millions, or billions) of individual vectors. Modern computers are also optimized to operate on matrices, providing significant performance gains compared to similar operations on a matrix vs. a series of vectors. The larger the data, the more tangible tangible and cost effective these performance gains become.

# In[13]:


aRow = np.array([[1,2]])  # notice the double brackets
bRow = np.array([[2,1]])

print('a shape:', a.shape) 
print('b shape:', b.shape)  
print('aRow shape:', aRow.shape)  # 1 row and 2 columns
print('bRow shape:', bRow.shape)  # also 1 row and 2 columns


# <span style="color:black">Using this structure does not change the operation behavior, but the outputs will have the same dimensions as the inputs (also with double brackets).

# In[14]:


aRow + bRow    #  results in 2D array representing 2D vector


# Once again compare `a` to `aRow` and note the number of brackets used in each.  In `aRow`, each set of inner brackets defines a new row of a matrix (in this example, just one row).

# In[15]:


a
aRow


# When you perform operations on these vectors, NumPy will convert the vector with the simpler structure to the more complex structure as long as the matrix has only one row and the same number of dimensions / values. Try this by adding `a` and `aRow`.

# In[16]:


a + aRow


# A 2x1 matrix has vertical and horizontal dimensions. As you have already seen, you can create a matrix with multiple rows and just a single column. As before, each set of inner brackets defines a new row in a matrix.

# In[17]:


aCol = np.array([[1],[2]])
aCol
print('shape:', aCol.shape)  # two rows and one column


# A **transpose** operation converts between horizontal and vertical 2D NumPy arrays. In general, the transpose flips the table around its diagonal values (values from top left corner of a matrix to bottom right corner of the matrix). In a column matrix, there is only one diagonal element. 
# 
# Also, **diagonal** elements have the same row and column index values. For example, in `aRow`, the value 1 has index (0,0) and, hence, is on diagonal. However, the value 2 has an index (1,0), and, hence, is not on diagonal, since index values are different.

# In[18]:


print('Transposed row as column: \n', aRow.T)
print('Transposed column as row: \n', aCol.T)


# Although not recommended, you can perform operations on simple vectors and matrices with **mixed** dimensions (meaning a 1x2 matrix and a 2x1 matrix, or either of those with a 2D vector). NumPy will try to convert the vectors to matrices of sufficient dimensionality to obey element-wise operation. 

# In[19]:


# 1x2 matrix + 2D vector

print("a and aRow")
a
aRow
print("********")
print("add aRow + a")
aRow + a # will treat the 2D vector like a 1x2 matrix, resulting in a 1x2 matrix


# In[20]:


# 2x1 matrix + 2D vector

print("a and aCol")
a
aCol
print("add aCol + a")
aCol + a # creates a 2x2 matrix 


# However, this can lead to unexpected behavior in your code and you should avoid it. For example, NumPy automatically ["broadcasts"](https://numpy.org/doc/stable/user/basics.broadcasting.html) (or "recycles") dimensions of a shorter array, but this may not be the intended mathematical behavior. In `aRow + aCol`, NumPy pads matrices till insufficient dimensions are added, without throwing an error. But is this what you really wanted mathematically?

# In[21]:


# 1x2 matrix + 2x1 matrix

aRow + aCol  # creates a 2x2 matrix


# Unexpected shapes of outputs (for example, adding a 1x2 and 2x1 matrix and getting a 2x2 matrix in response) can cause hard-to-debug issues across a large, complex code base. For best results, you should always make sure your matrices are the same shape **before** performing a linear algebra operation. 
# 
# You can convert a vector into a matrix by specifying where the additional dimension should be added with either `np.newaxis` or `None`. You can also use the NumPy reshape function and specify the desired matrix shape.

# In[22]:


a[np.newaxis, :]  # same as a[None, :], makes a row vector
a.reshape([1,2])  # same result as above
a[:, None]        # makes a column vector
a.reshape([2,1])  # same result as above


# NumPy arrays can be sliced using 0-based index notation, just like strings. However, the **concatenation** operation is different since `+` is already used for an element-wise addition. 

# In[ ]:


print("*****2D vector work*****")
a[0]                           # retrieve the element in 0th position
a[1]                           # retrieve the element in position 1
a[:1]                          # retrieve a subarray up to the specified position minus 1. Same as a[0:1]
a[1:]                          # retrieve a subarray starting at position 1 to the end of the array
np.concatenate([a,b])          # concatenate two vectors as a 1x4 matrix
np.concatenate([a,b], axis=0)  # same output as above
#np.concatenate([a,b], axis=1) # this operation will fail since vectors technically don't have "columns"
np.r_[a, b]                    # concatenate along the first axis


# In[ ]:


print("*****1x2 matrix work*****")
aRow[0][0]                     # retrieve the value in row 0, column 0
aRow[0][1]                     # retrieve the value in row 0, column 1
np.concatenate([aRow,bRow])    # concatenate two 1x2 matrices, resulting in a 2x2 matrix
np.concatenate([aRow,bRow], axis = 0) # same as above
np.concatenate([aRow,bRow], axis = 1) # results in a 1x4 matrix


# Note that once vectors have more than two values, you can no longer plot them in a 2D image, i.e., a 2-dimensional coordinate system.
# 
# <h3 style="color:black"> Dot (Inner) Products</h3>
# 
# Here you will learn about a very popular matrix operation known as **dot product**, also referred to as **inner product** or **sum product**. Dot products are frequently used to compute similarity metrics for vectors. Dot product works with two matrices that have matching "inner" values. For example, you could compute a dot product for a 3x5 matrix and a 5x4 matrix because the column value of the first matrix matches the row value in the second (they are both 5). However, unlike element-wise multiplication, the order of the matrices is important. You could not reverse and try to take the dot product of the 5x4 matrix and the 3x5 matrix, because the inner values (now 4 and 3) do not match. 
# 
# Dot product treats a column of values in the first matrix like a vector and multiplies it by a row of values in the second matrix (which is why the inner values have to match). When it has computed the products, it then stores the sum of those products as a value in the resulting matrix. For example, if the 0-index column in the first matrix had two values (1, 2) and the 0-index row in the second matrix had two values (3, 4), the product of those two "vectors" would be (3, 8). The value stored would be the sum of these values, or (11). 
# 
# The shape of the resulting matrix (the number of values stored) is determined by the outer shape values. For example, the final shape of the dot product of a 3x5 matrix and a 5x4 matrix would be a 3x4 matrix. 
# 
# In contrast, element-wise multiplication, also known as a **Hadamard product**, of $a$ and $b$ (two vectors with the same dimensionality) can be represented as $a \odot b$. Similarly, element-wise division can be represented as $a\oslash b$. Element-wise multiplication requires that each matrix be the same shape from the beginning; thus, you cannot perform an element-wise multiplication operation on a 3x5 matrix with a 5x4 matrix.

# In[ ]:


# element-wise operation examples 

a * b           # element-wise multiplication, a.k.a. Hadamard product
a * a * a * a   # a Hadamard product applied sequentially
a**4            # element-wise power
a / b           # element-wise division


# The sum of the outputs from element-wise multiplication of two vectors with equal dimensionality is equal to the dot product. This operation is denoted as $a\bullet b$ (with a large dot, not an asterisk) in mathematics or `a @ b` in Python.

# In[ ]:


sum(a * b)   # a sum of element-wise multiplication, i.e. sum product or inner product
a @ b        # the same sum product using dot product notation
#a @ aRow    # this operation will fail because the inner shape of a = 2, whereas aRow is 1x2
aRow @ a     # this operation will work because aRow's inner value = 2, which matches a
print("Note that the dimension of aRow @ a matches the outer values of 1 for aRow and <blank> for a")


# One application of a dot product is to check whether it is zero, in which case such pair of vectors are defined to be **orthogonal** (or **perpendicular**, i.e., with an angle of 90 degrees). For example:
# 
# 1. The vectors aligned with axes are orthogonal: $[0,a], [b,0]$, for any scalar values $a,b\ne 0$. 
# 1. $[0,1,1]\bullet[1,1,-1]=0$ (try drawing this vector).
# 1. Try drawing any two orthogonal vectors on paper and then compute their dot product.

# In[ ]:


np.array([0,1,1]) @ np.array([1,1,-1])


# ## Copying NumPy Arrays
# 
# A secondary but still important functionality of NumPy arrays (and Pandas DataFrames) is referencing and copying. This will be useful in handling objects throughout this course sequence.
# 
# In Python, basic data types (integers, floats) are copied by value, but more complex data types are copied by reference instead because it is faster. This means, for example, that the assignment `d = c` does not create two vectors but instead has both variables point to the same location in memory. 
# 
# Recall from your Python prerequisite: You can use [`np.copy`](https://numpy.org/doc/stable/reference/generated/numpy.copy.html) or [`c.copy`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html#numpy.ndarray.copy) or [`copy.copy`](https://docs.python.org/3/library/copy.html) to copy by value instead, which creates a new source.

# In[ ]:


c = np.array([1,1])
d = c                # caution: this is a copy by reference. d points to location of c
e = c.copy()         # this is a copy by value. a new source is created
c[0] = 99            # change the source variable's 0th element
d                    # d and c are still the same
e                    # e remains unchanged with the old value of c


# <hr style="border-top: 2px solid #606366; background: transparent;">
# 
# # **Optional Practice**
# 
# Now you will practice vector operations with a vector of zeros, $z$, and a vector of ones, $x$. Create these vectors in Python. 

# In[ ]:


z = np.zeros(5)   # create a 2D vector of zeros with 1D array
x = np.ones(5)    # create a 2D vector of ones with 1D array
print('z: ', z)
print('x: ', x)


# As you work through these tasks, check your answers by running your code in the *#check solution here* cell, to see if you’ve gotten the correct result. If you get stuck on a task, click the See **solution** drop-down to view the answer.
# 
# ## Task 1
# 
# Compute $z+x,\,\,\,z-x,\,\,\,10x,\,\,\,x/10,\,\,\,z\odot x,\,\,\,x\odot x,\,\,\,x\bullet x$
# 
# <b>Hint:</b> Check code above, where these operations are defined and used.

# In[ ]:


# check solution here


# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# z + x    # vector addition
# z - x    # vector subtraction
# 10 * x   # element-wise multiplication by a scalar
# x / 10   # element-wise division by a scalar
# z * x    # element-wise multiplication
# x * x    # element-wise multiplication
# x @ x    # dot product (or sum product)
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 2
# 
# Copy $z$ to $y$ by value, change $y_0$ to 2 (i.e., 0th element of vector $y$) and compute $10(y+x)$
# 
# <b>Hint:</b> Check code above where these operations are used.

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# y = z.copy()
# y[0] = 2
# 10 * (y + x)
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 3
# 
# Can you compute the answer manually, i.e., without using Python code?
# Note: The order of operations is left to right with no priority to `*` or `@` operations.
# 
#     x @ x * x
#     x * x @ x
#     
# <b>Hint:</b> Work out the algebra on a piece of paper and compare your result to outputs from running the code lines.

# In[ ]:


# check solution here after working these out on a piece of paper


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre> 
# Simply execute these lines to check your calculations:
# x @ x * x
# x * x @ x
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 4
# 
# Can you determine the answer manually, i.e., without using Python code? Note: Power operation has a priority.
# 
#     x ** x @ x * x
#     x * x @ x ** x
#     
# <b>Hint:</b> Work out the algebra on a piece of paper and compare your result to outputs from running the code lines.

# In[ ]:


# check solution here after working these out on a piece of paper


# 
# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#  Run the lines of code to see the solution. <pre>x ** x @ x * x
# x * x @ x ** x</pre>
# <hr>

# ## Task 5
# 
# Can you determine the answer manually, i.e., without using Python code?
# 
#     x @ x @ x
#     
# <b>Hint:</b> Is <code>x @ x </code> a scalar or a vector? Work out the algebra on a piece of paper and compare your result to outputs from running the code lines.

# In[ ]:


# check solution here after working these out on a piece of paper


# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
# The code <code>x @ x @ x</code> throws an error since <code>@</code> (dot product) expects "inner" values to match, which is true for the first <code>@</code>, but not for the second <code>@</code>. The first <code>x @ x</code> is an operation on two vectors with no "outer" values, therefore <code>x @ x</code> returns a scalar. Feel free to test this for yourself in a test cell with the following experiment:
#         
#         <code>value = x @ x</code>
#         <code>value.shape</code>
#         
# Therefore, when the second <code>@</code> is reached, there is no "inner" value of the previous dot product to match.
# <hr>

# ## Task 6
# 
# Compute $x / \sqrt{x\bullet x}$ and save results to $u$ variable. Notice that $\sqrt{x\bullet x}$ is a scalar, so this is just a scalar division (or multiplication by its reciprocal, which is still a scalar).
# 
# <b>Hint:</b> Check code above where we used these operations.
# 

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
#     <pre>
# xLen = (x @ x)**0.5  # Euclidean length of vector x
# u = x / xLen         # unit vector (of length 1) in the direction of x
# u
# (u @ u)**0.5         # confirm that the length of u is 1 (with minor precision error)
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 7
# 
# Convert $x$ to a row and column 2D NumPy arrays, `xRow` and `xCol`, respectively. Then, compute `xRow + xRow` and `xCol + xCol`.
# 
# <b>Hint:</b> <code>x[None, :]</code> slice adds one more array dimension in axis 0.
# 

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
# <pre>
# # Solution 1:
# xRow= x[None, :]
# xCol = x[:, None]
# xRow + xRow
# xCol + xCol
# # Solution 2:
# xRow = np.array([x])
# xCol = xRow.T
# xRow + xRow
# xCol + xCol
#     </pre>
#     </details> 
# </font>
# <hr>

# ## Task 8
# 
# Concatenate the arrays $u$ and $x$, then calculate the dot product of the result with itself.
# 
# <b>Hint:</b> <code>np.r_</code> or another concatenation function above may be helpful.
# 

# In[ ]:


# check solution here


# 
# <font color=#606366>
#     <details><summary><font color=#b31b1b>▶ </font>See <b>solution</b>.</summary>
# <pre>
# # Solution 1:
# np.r_[u,x] @ np.r_[u,x]
# # Solution 2:
# c = np.concatenate([u,x], axis=0)
# c @ c
#     </pre>
#     </details> 
# </font>
# <hr>

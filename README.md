# linear-algebra
implemented vector and matrix classes with reST-formatted docstrings in Python 3+

## General Layout

The Vector class imitates the m x 1 vector from linear algebra and
contains many useful functions for dealing and interacting with Vectors.
There are no getters and setters by design; the user should understand that
they are not allowed to change the attributes directly, and should instead use
the appropriate methods (ex. resize() or set()). They can access the component
attribute to get specific values, but they should not change it themselves.
    
    class Vector
    
    __init__(comp)          - takes in a list of components or a valid mx1 Matrix
    
    resize(length)          - while preserving current elements or filling with 0's, changes current vector length
    
    set(comp, index=-1)     - sets entire list at once or one specific index/value
    
    zero()                  - turns the current vector into a zero vector and returns it
    
    mag()                   - returns the magnitude of current vector
    
    normalize(change=False) - returns normalized current vector, if change=True, internal vector is updated
    
    same_normalized(other)  - returns True/False depending on equality of the two vectors once normalized
    
    dot(other)              - returns the dot product of th two vectors
    
    cross(other)            - returns the cross product of u x v (u is current vector, v is other)

    operator +              - returns sum of two vectors, component wise addition
    
    operator -              - returns difference of two vectors, component wise subtraction
    
    operator *              - alternate for dot product, or can use for vector scaling
    
    operator **             - returns original vector with its components raised to power
    
    operator ==             - checks to see if lists are equal
    
    to string method        - format: "<elem1, elem2, ...>"
    
    length = number of components in vector
    
    rows = same as length, used with cols for backwards compatibility with Matrix
    
    cols = 1 (num of columns)
    
The Matrix class imitates the matrix concept from linear algebra and allows
for different ways of dealing and interacting with matrices and vectors.
There are no getters and setters by design; the user should understand that
they are not allowed to change the attributes directly, and should instead use
the appropriate methods (ex. resize() or set()).
    
    class Matrix
    
    __init__(comp)          - takes in a list of components or a valid Vector
    
    resize(rows, cols)      - while preserving current elements or filling with 0's, changes current vector length
    
    set(comp, index=-None)  - sets entire list at once or one specific index/value (tuple or array as (row, col))
    
    zero()                  - replaces values in current matrix with all zeroes and returns it
    
    operator +              - returns sum of two matrices, component wise addition
    
    operator -              - returns difference of two matrices, component wise subtraction
    
    operator *              - matrix multiplication, matrix-vector product, scalar multiplication
    
    operator **             - returns original matrix with its components raised to power
    
    operator ==             - checks to see if internal lists are equal
    
    to string method        - format: "[row1\n row2\n row3\n ...]"  '\n ' is for spacing and alignment
    
    rows = number of rows in matrix
    
    cols = number of columns in matrix
    
## Tests

There is a file for writing tests. At the moment, it is only a series of assertions. In the future, I will update the testing file to use a more standardized version/library for testing.

## Docstrings/Documentation

There is thorough documentation throughout the library. Python library contains a help() function that could be used to view the docstrings for a specific class/method.

Class: help(classname)

Method: help(classname.methodname)

For example, help(Vector.resize) shows you ...

    resize(self, length)

        Re-sizes the vector to the specified length. If length is greater than

        current size, the new components are initialized with 0, otherwise if length

        is less than current size, the last few components are lost.

        :param length: new length of vector

        :type length: int

        :return: current vector, now resized

        :rtype: Vector

and vice versa. help(Vector) and help(Matrix) return major class overviews.

## Using library

Import the linear module into your library. The exact file with all of the necessary code is linear.py.

## Potential additions to library in future

### Vector Class

* Method for checking if current vector and other vector are orthogonal

* Support for plotting/graphing a certain amount of 2D maybe 3D vectors using matplotlib or a similar library

* ~~Cross products~~

### Matrix Class

* determinant

* Row Reducing Matrices

* Finding inverse of Matrices

* Transposing matrices

* Standalone function for fitting a straight line with least squares approximation

* General least squares method as part of matrices class






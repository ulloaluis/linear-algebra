# -*- coding: utf-8 -*-
"""
:date: 2018-07-30
:author: Luis Ulloa
:license: MIT-license

My implementation of a linear algebra library in Python. Consists of
a Vector and Matrix class that can be compatible with one another, as
well as useful methods for interacting with both concepts/classes.
"""

from math import pow, sqrt          # C implementation is best
from linear_lib.linear_tests import *


class Vector:
    """
    The Vector class imitates the m x 1 vector from linear algebra and
    contains many useful functions for dealing and interacting with Vectors.

    There are no getters and setters by design; the user should understand that
    they are not allowed to change the attributes directly, and should instead use
    the appropriate methods (ex. resize() or set()).

    Getting values directly from the vector can be achieved by accessing the
    component in comp using 0 based indexing.

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
    """

    def __init__(self, comp=[]):
        """
        Initializes the vector with either a list containing its components
        or an appropriate Matrix with mx1 dimensions. Defaults to an empty
        vector if not specified.

        :param comp: a list of the elements to be included in the vector;
                        the initial components, defaulted to an empty list.
                        could also be an appropriately sized Matrix
        :type comp: list, Matrix
        :return: none
        :raises: ValueError when Matrix is invalid size for Vector conversion
        """
        if isinstance(comp, Matrix):
            if comp.cols == 1:
                self.comp = [row[0] for row in comp.comp]
            else:
                raise ValueError("Cannot convert Matrix with greater than 1 column to Vector.")
        else:
            self.comp = comp             # user should never change comp directly; use set()

        self.length = len(self.comp)     # user should never change length directly; use resize()
        self.rows = self.length          # rows and cols included for backwards compatibility as
        self.cols = 1                    # a Matrix and for use in matrix-vector product

    def resize(self, length):
        """
        Re-sizes the vector to the specified length. If length is greater than
        current size, the new components are initialized with 0, otherwise if length
        is less than current size, the last few components are lost.

        :param length: new length of vector
        :type length: int
        :return: current vector, now resized
        :rtype: Vector
        """
        assert(length >= 0)                # no negative lengths
        dist = length - self.length

        if dist < 0:
            self.comp = self.comp[:dist]  # notice how this fails when dist = 0, but else correctly handles it
        else:
            self.comp = self.comp + [0]*dist

        self.length = length
        return self

    def set(self, comp, index=-1):
        """
        Set/change the values of the current vector. Can either pass in a new
        list to replace the internal list, or can specify an index in vector
        to change just that value, in which case comp can be a single value.
        No errors are thrown if user re-sizes the list.

        :param comp: list to replace whole vector or value to replace single component
        :param index: optional parameter that specifies the index of the value to be replaced
        :type comp: list, int, float
        :type index: int, float (that is whole ex. 1.0)
        :return: current vector, now updated
        :rtype: Vector
        """
        if index < 0:                   # default None and index=0 calls would conflict
            self.comp = comp
            self.length = self.rows = len(comp)
        else:
            assert(index < self.length)
            self.comp[index] = comp
        return self

    def zero(self):
        """
        Zeroes out the current vector by replacing each component with a 0.

        :return: returns current vector, which is now a zero vector
        :rtype: Vector
        """
        self.comp = [0]*self.length
        return self

    def mag(self):
        """
        Will get the magnitude of a vector.

        :return: the magnitude of a vector (sqrt(sum of components squared))
        :rtype: int, float
        """
        return sqrt(sum([pow(x, 2) for x in self.comp]))

    def normalize(self, change=False):
        """
        Normalizes a vector (acts on internal vector, does not take in a vector)

        :param change: if True, internal vector components are changed in addition
                          to returning vector
                 if False, vector says the same but normalized vector is returned;
                           default is false
        :type change: bool
        :return: another Vector but with the normalized components (False)
                 current Vector but with normalized components (True)
        :rtype: Vector
        """
        magnitude = self.mag()
        if magnitude == 0:      # already zero vector
            return self

        if change:
            for i in range(len(self.comp)):
                self.comp[i] /= magnitude
            return self
        else:
            return Vector([x / magnitude for x in self.comp])

    def same_normalized(self, other):
        """
        This function states whether the current vector is the same as other
        vector when normalized.

        :param other: other vector to be compared
        :type other: Vector
        :return: True if they have same normalized version, False otherwise
        :rtype: bool
        """
        return self.normalize() == other.normalize()

    def dot(self, other):
        """
        This function returns a scalar (number) value representing the dot
        product of the current vector and the other vector.

        :param other: the b vector in a dot b
        :type other: Vector
        :return: The dot product of the current vector and other vector.
        :rtype: int, float
        :raises: ValueError when vectors are not the same length
        """
        if len(self.comp) == len(other.comp):
            return sum([x * y for x, y in zip(self.comp, other.comp)])
        else:
            raise ValueError("Invalid vectors - must be of same length.")

    def cross(self, other):
        """
        For 3-dimensional vectors (3x1), this function allows you to take
        the cross product, which produces a vector i.e. orthogonal to both.

        :param other: 3D Vector (b in a X b)
        :return: Vector representing cross product of current and other
        :rtype: Vector
        """

        # Simplified version, after determinants: u is current vector v is other
        # u x v = (u2v3 - u3v2)i - (u1v3-u3v1)j + (u1v2-u2v1)k
        if self.length == 3 and other.length == 3:
            i_hat = self.comp[1]*other.comp[2] - self.comp[2]*other.comp[1]
            j_hat = -1 * (self.comp[0]*other.comp[2] - self.comp[2]*other.comp[0])
            k_hat = self.comp[0]*other.comp[1] - self.comp[1]*other.comp[0]
            return Vector([i_hat, j_hat, k_hat])
        else:
            raise ValueError("Invalid vectors - Can only take the cross product of 3D vectors.")

    def __add__(self, other):
        """
        Adding two vectors returns a vector with the respective components
        added together as expected. (does not affect this or other vector's
        components)

        :param other: the other vector to be added to current instance vector
        :type other: Vector
        :return: a vector with the resulting added components
        :rtype: Vector
        :raises: ValueError when vectors are not the same length
        """
        if len(self.comp) == len(other.comp):
            return Vector([x+y for x, y in zip(self.comp, other.comp)])
        else:
            raise ValueError("Invalid vectors - must be of same length.")

    def __sub__(self, other):
        """
       Subtracting two vectors returns a vector with the respective components
       subtracted. "current - other" is formatting. (this does not affect this
       or other vector's components)

       :param other: the other vector which is subtracting from the current vector
       :type other: Vector
       :return: a vector with the resulting subtracted components
       :rtype: Vector
       :raises: ValueError when vectors are not the same length
       """
        if len(self.comp) == len(other.comp):
            return Vector([x-y for x, y in zip(self.comp, other.comp)])
        else:
            raise ValueError("Invalid vectors - must be of same length.")

    def __mul__(self, other):
        """
        Multiplies the two vectors together; same functionality as calling the
        dot() function for dot product of current and other. Could also scale
        each component by a number

        :param other: the other vector
        :type other: Vector, integer, float
        :return: number value representing dot product of both vectors
        :rtype: int, float
        :raises: ValueError when vectors are not the same length
        """
        if isinstance(other, int) or isinstance(other, float):
            return Vector([x * other for x in self.comp])
        elif len(self.comp) == len(other.comp):
            return self.dot(other)
        else:
            raise ValueError("Invalid vectors - must be of same length.")

    def __eq__(self, other):
        """
        If two vectors have the same components, then they are equal. If the
        lists are not the same length, will always be False with no error thrown.

        :param other: other vector being tested for equality
        :type other: Vector
        :return: True or False based on equality
        :rtype: bool
        """
        return self.comp == other.comp  # compares lists

    def __pow__(self, power, modulo=None):
        """
        Allows you to raise each of the components of the current vector to a power.

        :param power: value to raise each component to
        :param modulo: optional parameter that applies the modulus operator to each result
        :type power: int, float
        :type modulo: int, float
        :return: a vector containing the appropriately scaled components
        :rtype: Vector
        """

        return Vector([pow(x, power) % modulo if modulo else pow(x, power) for x in self.comp])

    def __str__(self):
        """
        Converts vector to string by placing the components a, b , c, ... into arrow
        brackets, as such <a, b, c, ...>.  Empty arrays return <empty> for clarity.

        :return: a string detailing contents of vector with the format <a, b, c, ...> or <empty>
        :rtype: str
        """
        if self.length == 0:
            return "<empty>"
        vec = "<"
        for elem in self.comp:
            vec += str(elem) + ", "
        return vec[:-2] + ">"  # remove additional ", " and close


class Matrix:
    """
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
    det()                   - takes the determinant of the current NxN matrix
    transpose()             - transposes the current mxn matrix to an nxm matrix (1st row becomes 1st col, etc.)
    operator +              - returns sum of two matrices, component wise addition
    operator -              - returns difference of two matrices, component wise subtraction
    operator *              - matrix multiplication, matrix-vector product, scalar multiplication
    operator **             - returns original matrix with its components raised to power
    operator ==             - checks to see if internal lists are equal
    to string method        - format: "[row1\n row2\n row3\n ...]"  '\n ' is for spacing and alignment

    rows = number of rows in matrix
    cols = number of columns in matrix
    """
    def __init__(self, comp=[]):
        """
        Initializes the matrix to the specified format. Default is an empty 0x0 matrix

        Note: It is up to the user to pass well-formed matrices, that is, two different
                rows cannot be different lengths, etc.

        :param comp: list of lists where each individual list represents a row,
                     similar to how numpy implements arrays; could also be a vector
        :type comp: list, Vector
        :return: none
        """

        if isinstance(comp, Vector):
            self.comp = [[row] for row in comp.comp]  # m x 1 Vector --> m rows
        else:
            self.comp = comp                          # list

        self.rows = len(self.comp)          # User should never be able to change instance
        if self.rows != 0:                  # variables directly, use an appropriate method
            self.cols = len(self.comp[0])
        else:
            self.cols = 0                   # if rows = 0, then no columns by default

    def resize(self, rows, cols):
        """
        Re-sizes the current matrix to the specified dimensions, rows x cols.
        Previous elements are left in place, if size is increased then new
        locations are filled with values of 0.

        :param rows: new row size
        :param cols: new column size
        :type rows: int, float
        :type cols: int, float
        :return: current matrix after resizing
        :rtype: Matrix
        """
        assert(rows >= 0 and cols >= 0)     # no negative dimensions allowed

        dist_rows = rows - self.rows    # 4-3 = 1
        dist_cols = cols - self.cols    # 4-2 = 2

        if dist_rows < 0:
            self.comp = self.comp[:dist_rows]
        else:
            for i in range(dist_rows):
                self.comp.append([0]*self.cols)  # update rows but don't have varying number of columns for each row

        if dist_cols < 0:                        # go through and shape columns now
            for i in range(rows):
                self.comp[i] = self.comp[i][:dist_cols]
        else:
            for i in range(rows):
                self.comp[i] += [0]*dist_cols

        self.rows = rows
        self.cols = cols
        return self

    def set(self, comp, index=None):
        """
        Set/change the current matrix. If index is not specified, then comp should
        be a list of lists detailing a new matrix. Otherwise, comp should be the
        integer/float value that goes in the specified index (row, column) tuple.

        :param comp: list of lists to replace matrix entirely, or single value
                     to replace a specific location in matrix
        :param index: optional tuple/list with (row, column) of value to be replaced
        :type comp: list of lists, int, float
        :type index: tuple, list
        :return: self, after edits are made
        :rtype: Matrix
        """

        if not index:
            assert(isinstance(comp, list))
            self.comp = comp
            self.rows = len(comp)
            if self.rows != 0:
                self.cols = len(comp[0])
            else:
                self.cols = 0
        else:
            assert(isinstance(comp, int) or isinstance(comp, float))
            self.comp[index[0]][index[1]] = comp
        return self

    def zero(self):
        """
        Zeroes out the current matrix by replacing every element with a zero.

        :return: The current matrix, but updated to be the zero matrix.
        """
        self.comp = [[0]*self.cols for _ in range(self.rows)]
        return self

    def det(self):
        """
        Returns the determinant of an nxn matrix that is at least a 2x2. (recursive)

        :return: the determinant of the current matrix
        :rtype: int, float
        """

        if self.rows != self.cols:
            raise ValueError("Invalid matrix - only N x N matrices supported.")

        # base case -> 2 by 2
        if self.rows == 2 and self.cols == 2:  # ad - bc
            return self.comp[0][0] * self.comp[1][1] - self.comp[0][1] * self.comp[1][0]

        # going along top, along first row (not optimized to find best path)
        top_row = self.comp[0]
        determinant = 0
        for col_i in range(len(top_row)):
            # don't include in same row or column
            new_matrix = self.comp[1:]            # remove top row
            for r in range(len(new_matrix)):      # remove this column from each row
                new_matrix[r] = new_matrix[r][:col_i] + new_matrix[r][col_i + 1:]

            constant = top_row[col_i]
            if col_i % 2 == 1:
                constant *= -1  # every other constant is negative

            determinant += constant * Matrix(new_matrix).det()
        return determinant

    def transpose(self):
        """
        This function will return the transpose of the current matrix. (A -> A^T)
        "First row becomes first column, second row becomes second column, etc."

        :return: Transposed matrix
        :rtype: Matrix
        """

        new_matrix = [[] for _ in range(self.cols)]     # num rows becomes num cols
        for r in range(self.rows):                      # go through each row
            for c in range(self.cols):                  # for each col in row
                new_matrix[c].append(self.comp[r][c])   # disperse columns into new rows

        return Matrix(new_matrix)

    def __add__(self, other):
        """
        Adds two matrices and returns a matrix with the respective components
        added together as expected.

        :param other: the other matrix to be added to current instance matrix
        :type other: Matrix
        :return: a matrix with the resulting added components
        :rtype: Matrix
        :raises: ValueError when matrices do not have same dimensions
        """
        new_comp = []
        if self.rows == other.rows and self.cols == other.cols:
            for x, y in zip(self.comp, other.comp):
                new_comp.append([a + b for a, b in zip(x, y)])  # adding done in list comprehension
            return Matrix(new_comp)
        else:
            raise ValueError("Size mismatch, both matrices must have the same number of rows and columns.")

    def __sub__(self, other):
        """
       Subtracting two matrices returns a matrix with the respective components
       subtracted. "current - other" is formatting.

       :param other: the other matrix which is subtracting from the current matrix
       :type other: Matrix
       :return: a matrix with the resulting subtracted components
       :rtype: Matrix
       :raises: ValueError when matrices do not have same dimensions
       """
        new_comp = []
        if self.rows == other.rows and self.cols == other.cols:
            for x, y in zip(self.comp, other.comp):
                new_comp.append([a - b for a, b in zip(x, y)])  # subtracting done in list comprehension
            return Matrix(new_comp)
        else:
            raise ValueError("Size mismatch, both matrices must have the same number of rows and columns.")

    def __mul__(self, other):
        """
        Multiplies the two matrices together; aka Matrix Multiplication.
        Matrix-Vector product is also possible using the Vector class, though
        this method works for a mx1 matrix as well. Also configured to work with
        normal application of multiplying a scalar to a matrix.

        Notes: Approach is to take the dot product of each row of current matrix
                with each column of other matrix/vector. Since you typically write
                "Ax" where A is the matrix and x is the vector, this syntax should
                be adhered to when attempting matrix multiplication with these classes.

        :param other: the other matrix or vector, could also be an int or float for scaling
        :type other: Matrix, int, float
        :return: the resulting matrix
        :rtype: Matrix
        :raises: ValueError when there's a matrix multiplication size mismatch ([mxn]*[nxp]=[mxp])
        """
        new_matrix = []
        if isinstance(other, int) or isinstance(other, float):
            for row in self.comp:
                new_matrix.append([elem * other for elem in row])
            return Matrix(new_matrix)
        elif self.cols == other.rows:    # [m x n] * [n x p] = [m x p] i.e. [self.rows x other.cols] matrix
            other_cols = []
            for i in range(other.cols):  # extract columns from rows
                other_cols.append([row[i] if isinstance(other, Matrix) else row for row in other.comp])
            for row_me in self.comp:
                new_row = []
                for col_other in other_cols:
                    new_row.append(Vector(row_me) * Vector(col_other))   # Dot product of vectors
                new_matrix.append(new_row)
            return Vector([row[0] for row in new_matrix]) if other.cols == 1 else Matrix(new_matrix)
        else:
            raise ValueError("Size mismatch; [m x n] * [n x p] = [m x p] matrix")

    def __eq__(self, other):
        """
        If two matrices have the same components, then they are equal. If the
        lists are not the same length, will always be False with no error thrown.

        :param other: other matrix being tested for equality
        :type other: Matrix
        :return: True or False based on equality
        :rtype: bool
        """
        return self.comp == other.comp  # compares lists

    def __pow__(self, power, modulo=None):
        """
        Allows you to raise a matrix to a power, that is, each of the
        components of the current matrix is raised to a power. Can use
        power 0 to fill the current matrix with all 1s.

        :param power: value to raise each component to
        :param modulo: optional parameter that applies the modulus operator to each result
        :type power: int, float
        :type modulo: int, float
        :return: a matrix containing the appropriately scaled components
        :rtype: Matrix
        """
        new_comp = []
        for row in self.comp:
            new_row = []
            for elem in row:
                if modulo:
                    elem = elem % modulo
                new_row.append(pow(elem, power))
            new_comp.append(new_row)
        return Matrix(new_comp)

    def __str__(self):
        """
        String representation of matrix is each row separated by new line
        characters. This is done so that when printed it resembles a normal
        matrix as closely as possible.

        :return: string representation of current matrix
        :rtype: str
        """
        return "[" + '\n '.join([str(row) for row in self.comp]) + "]"  # intentional space after \n for alignment


if __name__ == "__main__":
    test()

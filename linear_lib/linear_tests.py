# -*- coding: utf-8 -*-
"""
:date: 2018-07-30
:author: Luis Ulloa
:license: MIT-license

Contains tests for linear algebra library, which contains a Vector
and Matrix class that are compatible with one another.

Note: Will re-write this test file using the proper Python
      testing library at a later time.
"""

from linear_lib.linear import *


def test_vector():
    x = Vector([1, 2, 3, 4, 5])
    y = Vector([2, 2, 2, 2, 2])
    assert(x.length == 5)
    assert(y.length == 5)
    assert(x*y == 30)
    assert(x+y == Vector([3, 4, 5, 6, 7]))
    assert(x-y == Vector([-1, 0, 1, 2, 3]))
    assert(x**2 == Vector([1, 4, 9, 16, 25]))
    assert(y**2 == Vector([4, 4, 4, 4, 4]))
    assert(str(x) == "<1, 2, 3, 4, 5>")
    assert(str(y) == "<2, 2, 2, 2, 2>")
    assert(not x.same_normalized(y))
    assert(y.mag() == sqrt(20))
    assert(x.zero() == Vector([0, 0, 0, 0, 0]))
    assert(x.normalize() is x)                      # normalizing zero vector has no effect
    assert(x.normalize(True) is x)                  # normalizing zero vector has no effect
    assert(x.set([1, 2, 3, 4, 5]) == Vector([1, 2, 3, 4, 5]))
    assert(x == Vector([1, 2, 3, 4, 5]))
    assert(x.set(3, 0) == Vector([3, 2, 3, 4, 5]))
    assert(x.set(1, x.length-1) == Vector([3, 2, 3, 4, 1]))
    assert(x.resize(3) == Vector([3, 2, 3]))
    assert(x.resize(5) == Vector([3, 2, 3, 0, 0]))
    assert(x.set([2, 2, 2, 2, 2]) == y)
    assert(x.same_normalized(y))
    assert(x.set(3, 0) == Vector([3, 2, 2, 2, 2]) and y.set(3, y.length - 1) == Vector([2, 2, 2, 2, 3]))
    assert(x.mag() == y.mag())
    assert(x.normalize(True).mag() == 1.0 and y.normalize(True).mag() == 1.0)
    assert(x*y == y*x)
    assert(x+y == y+x)
    assert(x.resize(0) == Vector())
    assert(str(x) == "<empty>")
    assert(x.set([1, 2, 3]) == Vector([1, 2, 3]) and x.length == 3)

    x = Vector([1, 2, 3])
    y = Vector([2, 2, 2])
    assert(x.cross(y) == Vector([-2, 4, -2]))

    assert(Vector([1, 0, 0]).cross(Vector([0, 1, 0])) == Vector([0, 0, 1]))  # i x j = k


def test_matrix():
    x = Matrix([[1, 2, 3],
                [3, 4, 5],
                [6, 7, 8]])
    y = Matrix([[4, 3, 6],
                [2, 1, 5],
                [1, 2, 1]])
    assert(str(x) == "[[1, 2, 3]\n [3, 4, 5]\n [6, 7, 8]]")
    assert(str(y) == "[[4, 3, 6]\n [2, 1, 5]\n [1, 2, 1]]")
    assert(x.rows == 3 and x.cols == 3 and y.rows == 3 and y.cols == 3)
    assert(x+y == Matrix([[5, 5, 9], [5, 5, 10], [7, 9, 9]]))
    assert(x-y == Matrix([[-3, -1, -3], [1, 3, 0], [5, 5, 7]]))
    assert(x**2 == Matrix([[1, 4, 9], [9, 16, 25], [36, 49, 64]]))
    assert(x*y == Matrix([[11, 11, 19], [25, 23, 43], [46, 41, 79]]))
    assert(y*x == Matrix([[49, 62, 75], [35, 43, 51], [13, 17, 21]]))
    assert(x.resize(2, 2) == Matrix([[1, 2], [3, 4]]))
    assert(x.resize(3, 3) == Matrix([[1, 2, 0], [3, 4, 0], [0, 0, 0]]))
    assert(x.set(([[1, 2, 3], [3, 4, 5], [6, 7, 8]])) == Matrix([[1, 2, 3], [3, 4, 5], [6, 7, 8]]))
    assert(x.set(500, (1, 1))) == Matrix([[1, 2, 3], [3, 500, 5], [6, 7, 8]])
    assert(x.zero() == Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    assert(x.resize(0, 0) == Matrix())
    assert(x.set([[1, 2], [2, 3]]) == Matrix([[1, 2], [2, 3]]) and x.rows == 2 and x.cols == 2)

    x = Matrix([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    assert(x.det() == 0)

    x = Matrix([[3, 2],
                [6, 4]])

    assert(x.det() == 0)

    x = x.set([[1, 2, 3], [7, 5, 3], [2, -4, 1]])
    assert(x.det() == -99)
    assert(x.set([[1, 3, 5, 9], [1, 3, 1, 7], [4, 3, 9, 7], [5, 2, 0, 9]]).det() == -376) # 4x4
    assert(x.set([[1, 7, 9, 2, 5], [3, 2, 3, 4, 6], [1, 2, 3, 4, 5], [4, 4, 6, 0, 9], [3, 3, 1, 2, 1]]).det() == 624)
    assert(x.set([[0, 0], [0, 0]]).det() == 0)
    assert(x.set([[0, 0, 0], [3, 9, 7], [6, 6, 4]]).det() == 0)


def test_class_overlaps():
    # converting from Vector to Matrix
    x = Vector([1, 2, 3, 4, 5])
    assert(Matrix(x) == Matrix([[1], [2], [3], [4], [5]]))
    assert(Vector(Matrix(x)*3) == x*3)

    # converting from appropriate Matrix to Vector
    y = Matrix([[1], [2], [3], [4], [5]])
    assert(Vector(y) == Vector([1, 2, 3, 4, 5]))
    assert(Matrix(Vector(y)**3) == y**3)

    # matrix vector products
    a = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x = Vector([1, 2, 3])
    assert(a*x == Vector([14, 32, 50]))

    aa = Matrix([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]])
    xx = Vector([1.1, 2.2, 3.3])
    assert(aa*xx == Vector([16.939999999999998, 38.72, 60.50000000000001]))


def test():
    test_vector()
    test_matrix()
    test_class_overlaps()

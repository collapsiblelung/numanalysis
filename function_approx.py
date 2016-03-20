# Nicole Hunkeapillar
# This program uses an orthonormal basis to approximate functions
# on the interval [-1, 1]. Let x be a column vector of 101 equally-
# spaced points in the interval. Let f(x) be a column vector y = f(x)
# of the corresponding y-values.

import numpy
import matplotlib.pyplot as plt
import numpy.linalg as la

# construct matrix of x values and matrix of y values
# we use the function y = sin(pi*x)
x = numpy.linspace(-1.0, 1.0, num=101)
y = numpy.sin(numpy.pi*x)

# construct each column of P, the Vandermonde matrix
p1 = numpy.zeros(101)
p1.fill(1)
p2 = x
p3 = numpy.square(p2)
p4 = numpy.power(p2, 3)
p5 = numpy.power(p2, 4)

# combine columns of P to form a matrix
P = numpy.matrix([p1, p2, p3, p4, p5])
P = P.T

# plot columns of p against x
plt.plot(x, p1, 'r--', x, p2, 'b--', x, p3, 'g--', x, p4, 'm--', x, p5, 'y--')
plt.ylabel('monomial basis functions')
plt.xlabel('x values')
plt.show()

# compute Q using the QR factorization 
Q, R = la.qr(P)

# extract columns of Q
q1 = Q[:, [0]]
q2 = Q[:, [1]]
q3 = Q[:, [2]]
q4 = Q[:, [3]]
q5 = Q[:, [4]]

# plot columns of q against x
plt.plot(x, q1.flat, 'r--', x, q2.flat, 'b--', x, q3.flat, 'g--', x, q4.flat, 'm--', x, q5.flat, 'y--')
plt.ylabel('QR Algorithm')
plt.xlabel('x values')
plt.show()

# compute projection matrix onto the column space of Q
proj = numpy.dot(Q, la.inv(numpy.dot(Q.T, Q)))
proj = numpy.dot(proj, Q.T)

# compute the projection y1 of y onto range(Q)
y1 = numpy.dot(proj, y).T

# plot y and y1 against x
plt.plot(x, y.flat, 'b-*', x, y1.flat, 'g--')
plt.ylabel('y & y1')
plt.xlabel('x values')
plt.legend(('orginal vector','projection'), loc='best')
plt.show()

# compute the error
e = y - y1.flat

# show that the error is orthogonal to each column pi 
# for i = 1, ..., 5
print ' '
print 'Dot Product of Error Values with p1 - p5'
print numpy.dot(e, p1)
print numpy.dot(e, p2)
print numpy.dot(e, p3)
print numpy.dot(e, p4)
print numpy.dot(e, p5)
print '--> approximately zero, which means e is orthogonal to pi'
print 'for i = 1, 5'
print ' '

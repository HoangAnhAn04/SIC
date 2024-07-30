import numpy as np
A = np.array([[1,3], [5,7]])
B = np.array([[4,-6], [-8,12]])
detA = np.linalg.det(A)
Ainv = np.linalg.inv(A)
detB = np.linalg.det(B)
print("Determinant of A=", detA)
print("Inverse of A=", Ainv)
print("Determinant of B=", detB)

## 2. Systems of equations as matrices ##

import numpy as np

# Set the dtype to float to do float math with the numbers.
matrix = np.asarray([
    [2, 1, 25],
    [3, 2, 40]  
], dtype=np.float32)

print(matrix)
matrix[0,:] = matrix[0,:] * 2
matrix[0,:] = matrix[0,:] - matrix[1,:]
matrix[1,:] = matrix[1,:] - 3*matrix[0,:]
matrix[1,:] = matrix[1,:] / 2
print(matrix)

## 4. Solving more complex equations ##

import numpy as np

matrix = np.asarray([
    [1, 2, 0, 7],
    [0, 3, 3, 11],
    [1, 2, 2, 11]
], dtype=np.float32)

print(matrix)
print("\n")
matrix[2,:] = matrix[2,:] - matrix[0,:]
matrix[2,:] = matrix[2,:] / 2
matrix[1,:] = matrix[1,:] / 3
matrix[1,:] = matrix[1,:] - matrix[2,:]
matrix[0,:] = matrix[0,:] - 2*matrix[1,:]
print(matrix)

## 5. Echelon form ##

matrix = np.asarray([
    [0, 0, 0, 7],
    [0, 0, 1, 11],
    [1, 2, 2, 11],
    [0, 5, 5, 1]
], dtype=np.float32)

# Swap the first and the third rows - first swap
matrix[[0,2]] = matrix[[2,0]]

temp = matrix.copy()
print(temp)
matrix[1,:] = temp[3,:]
matrix[2,:] = temp[1,:]
matrix[3,:] = temp[2,:]
print(matrix)

## 6. Reduced row echelon form ##

A = np.asarray([
        [0, 2, 1, 5],
        [1, 2, 1, 8],
        [3, 0, 1, 10],
        ], dtype=np.float32)

# First, we'll swap the second row with the first to get a non-zero coefficient in the first column
A[[0,1]] = A[[1,0]]

# The leading coefficient is already 1, so there's no need to divide
# Now, we need to make sure that our 1 coefficient is the only coefficient in its column
# We have to subtract three times the first row from the third row
A[2] -= 3 * A[0]

# Now, we move to row 2
# We divide by 2 to get a one as the leading coefficient
A[1] /= 2

# We subtract 2 times the second row from the first to get rid of
# the second column coefficient in the first row
A[0] -= 2 * A[1]

# And we'll add 6 times the second row to the third to eliminate the leading coefficient there
A[2] += 6 * A[1]

# Now, we can move to the third row where the leading coefficient is already 1
# We just need to subtract half of the third from the second
A[1] -= 0.5 * A[2]

# We're finished, and our system is solved!
print(A)

## 7. Inconsistency ##

A = np.asarray([
    [10, 5, 20, 60],
    [3, 1, 0, 11],
    [8, 2, 2, 30],
    [0, 4, 5, 13]
], dtype=np.float32)

B = np.asarray([
    [5, -1, 3, 14],
    [0, 1, 2, 8],
    [0, -2, 5, 1],
    [0, 0, 6, 6]
], dtype=np.float32)

A[2,:] = A[2,:] - (A[2,0]/A[0,0])*A[0,:]
A[1,:] = A[1,:] - (A[1,0]/A[0,0])*A[0,:]
A[3,:] = A[3,:] - (A[3,1]/A[1,1])*A[1,:]
A[2,:] = A[2,:] - (A[2,1]/A[1,1])*A[1,:]
A[3,:] = A[3,:] - (A[3,2]/A[2,2])*A[2,:]

A[0,:] = A[0,:] - (A[0,1]/A[1,1])*A[1,:]
A[0,:] = A[0,:] - (A[0,2]/A[2,2])*A[2,:]
A[1,:] = A[1,:] - (A[1,2]/A[2,2])*A[2,:]
A[0,:] = A[0,:] / A[0,0]
A[1,:] = A[1,:] / A[1,1]
A[2,:] = A[2,:] / A[2,2]
A_consistent = True

B[2,:] = B[2,:] - (B[2,1]/B[1,1])*B[1,:]
B[3,:] = B[3,:] - (B[3,2]/B[2,2])*B[2,:]
B_consistent = False
print(B)

## 8. Infinite solutions ##

A = np.asarray([
        [2, 4, 8, 20],
        [4, 8, 16, 40],
        [20, 5, 5, 10]
], dtype=np.float32)

B = np.asarray([
        [1, 1, 1, 4],
        [3, -2, 5, 8],
        [8, -4, 5, 10]
        ], dtype=np.float32)

A[0,:] = A[0,:] / A[0,0]
A[1,:] = A[1,:] / A[1,0]
A[2,:] = A[2,:] / A[2,0]
A_infinite = True

B[0,:] = B[0,:] / B[0,0]
B[1,:] = B[1,:] / B[1,0]
B[2,:] = B[2,:] / B[2,0]
B[2,:] = B[2,:] - B[0,:]
B[1,:] = B[1,:] - B[0,:]
B[2,:] = B[2,:] - (B[2,1]/B[1,1])*B[1,:]
B[0,:] = B[0,:] - (B[0,1]/B[1,1])*B[1,:]
B[0,:] = B[0,:] - (B[0,2]/B[2,2])*B[2,:]
B[1,:] = B[1,:] - (B[1,2]/B[2,2])*B[2,:]
B[1,:] = B[1,:] / B[1,1]
B[2,:] = B[2,:] / B[2,2]
B_infinite = False
print(B)
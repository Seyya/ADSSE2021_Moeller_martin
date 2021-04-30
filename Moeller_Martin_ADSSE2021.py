import numpy as np
import tracemalloc as malloc
import timeit

n = ArrSizeRow = 64
m = ArrSizeColumn = 64
ArrA = np.random.randint(50, size=(n,m))
ArrB = np.random.randint(50, size=(n,m))

#Strassen matrix multiplication!
def strassenSplit(matrix):
    
    #Spltting square matrix (nxn-matrix) into four (quarters)
    
    row, col = matrix.shape
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]
  
def algoStrassen(ArrA, ArrB): 

    #Using divideand conquer recursively allows it to compute the matrix product

  
    # If size of matrix is 1x1
    if len(ArrA) == 1:
        return ArrA * ArrB
  
 
    # Splits matrices into quadrants recursively, until base case
    a, b, c, d = strassenSplit(ArrA)
    e, f, g, h = strassenSplit(ArrB)

    # Recursively computes the seven products
    p1 = algoStrassen(a, f - h)  
    p2 = algoStrassen(a + b, h)        
    p3 = algoStrassen(c + d, e)        
    p4 = algoStrassen(d, g - e)        
    p5 = algoStrassen(a + d, e + h)        
    p6 = algoStrassen(b - d, g + h)  
    p7 = algoStrassen(a - c, e + f)  
  
    # Computation for final matrix C
    c11 = p5 + p4 - p2 + p6  
    c12 = p1 + p2           
    c21 = p3 + p4            
    c22 = p1 + p5 - p3 - p7  

    #Combining the final components of C into one single matrix by stacking
    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22)))) 
    
    return c

def NaiveAlgo(ArrA,ArrB):
    NaiveResult = np.random.randint(1, size=(n,m))
    for i in range(len(ArrA)):
    
        # iterating by coloum by B 
        for j in range(len(ArrB[0])):
  
            # iterating by rows of B
            for k in range(len(ArrB)):
                NaiveResult[i][j] += ArrA[i][k] * ArrB[k][j]
    return NaiveResult

#Start timer to calculate execution time"
start = timeit.default_timer()

#Malloc to trace memory usage#
#malloc.start()


#Run this code to use Strassen's algorithm
StrassenResult = algoStrassen(ArrA,ArrB)

#Run this code to use Naïve algoritm
#naiveResult = NaiveAlgo(ArrA,ArrB)

#Stop time to be able to calculate execution time
stop = timeit.default_timer()
print("Time", stop-start)

#Calculated memory in Kbs **6 for Mbs

#Traced memory usage when running code"
#print("memory", malloc.get_traced_memory()[1]/10**3)
#malloc.stop()

#print("StrassenResult \n", StrassenResult)
#print("ResultNaive \n", NaiveAlgo(ArrA,ArrB))


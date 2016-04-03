import numpy as np
def lcs(a,b):
    m = len(a)
    n = len(b)
    c = np.zeros((m+1,n+1))
    for i in range(1,m+1):
        for j in range(1,n+1):
        	if(a[i-1] == b[j-1]):
        		c[i][j] = c[i-1][j-1]+1
        	else:
        		c[i][j] = max(c[i][j-1], c[i-1][j])
    return c[m][n]


if __name__ == '__main__':
	a = [0,1,5,7,9,15,18,27,28]
	b = [1,5,6,15,16,17,18,25,26,27]
	print lcs(a,b)
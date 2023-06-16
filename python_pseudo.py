

from copy import deepcopy
import math
import cmath
import random
from datetime import datetime
import numpy as np
def get_samples(start, end,sample_size):
    samples = [start + (end-start)*random.random() for _ in range(sample_size)]
    return sum(samples)





def euler(N,k):

    return np.exp((-k*math.pi*2j)/N)



def DFT(X,inverse=False):
    RESULT = []
    N = len(X)
    sign =1
    if inverse:
        sign = -1
    for k in range(N):
        X_k = 0 
        for n in range(N):
            X_k = X_k +  euler(N,sign*-k*n)*X[n]
        if inverse:
            RESULT.append(X_k/N)
        else:
            RESULT.append(X_k)
    return RESULT
            

def FFT(X,inverse=False):
    '''Radix 2 fast fourier transform'''
    N = len(X)
    sign = 1
    if inverse:
        sign = -1
    assert N%2 == 0, "Must be poewers of 2"
    RESULT = []
    if N ==0:
        return []
    elif N == 1:
        RESULT =  [X[0]] 
    elif N == 2:
        RESULT =  [X[0]+X[1],X[0]-X[1]]
    else:
        X_EVEN = FFT(X[::2],inverse=inverse)
        X_ODD = FFT(X[1::2],inverse=inverse)
        X_PLUS = []
        X_NEG = []
        for i in range(N//2):
            X_PLUS.append(X_EVEN[i]+euler(N,sign*i)*X_ODD[i])
            X_NEG.append(X_EVEN[i]-euler(N,sign*i)*X_ODD[i])
        RESULT =  X_PLUS+X_NEG
    if inverse:
        RESULT = [r/2 for r in RESULT]
    return RESULT



def pad_zeros(X,L=0):
    N = len(X)
    RESULT = [j for j in X]
    for i in range(N,N+L):
            RESULT.append(0)
    N = len(RESULT)
    for i in range(N,2*N):
        if i^(i-1) > i:
            return RESULT
        else:
            RESULT.append(0)
    return RESULT
        

def convolution(X,Y):
    assert len(X) ==len(Y),"The convolution only works if they are the same lenght"
    N = len(X)
    X = pad_zeros(X,N)
    Y = pad_zeros(Y,N)
    for i in range(N):
        Y[-i] = Y[i]
    NEW_N = len(X)
    FX = FFT(X)
    FY = FFT(Y)
    FXY = [FX[i]*FY[i] for i in range(NEW_N)]
    RESULT = FFT(FXY,inverse=True)
    return RESULT[:N]


        

    
def bluestein_algorithm(X,inverse=False):
    '''
    FFT if its not radix-2
    '''      
    N = len(X)
    sign,M = 1,1
    if inverse:
        sign,M = -1,1/N
    A = [X[i]*euler(N=2*N,k=sign*i*i) for i in range(N)]
    B = [euler(N=2*N,k=sign*-i*i) for i in range(N)]
    CONOVOLVED = convolution(A,B)

    return [M*CONOVOLVED[i]*euler(N=2*N,k=sign*i*i) for i in range(N)]

def ALL_BASE_FFT(X,inverse=False):
    N = len(X)
    if N^(N-1) > N:
        return FFT(X=X,inverse=inverse)
    else:
        return bluestein_algorithm(X=X,inverse=inverse)

time_before = datetime.now()
BIG = 1333
samples = [get_samples(start=-0,end=1,sample_size=100) for _ in range(BIG)]

time_after_samples = datetime.now()
print(f'the time taken generating samples is is:{time_after_samples-time_before}')
print(samples)

dft_samples = deepcopy(samples)
dft_before  = datetime.now()
dft_transform = DFT(dft_samples)
dft_after = datetime.now()
print(f'the time taken dft samples  is:{dft_after-dft_before}')

fft_samples = deepcopy(samples)
fft_before  = datetime.now()
fft_transform = ALL_BASE_FFT(fft_samples)
fft_after = datetime.now()
print(f'the time taken fft samples  is:{fft_after-fft_before}')

numpy_fft = np.fft.fft(np.array(samples))
with open('output.csv','w') as file:
    for i in range(BIG):
        file.write(f'{dft_transform[i]},{fft_transform[i]} ,{numpy_fft[i]}\n')

numpy_og = np.fft.ifft(numpy_fft)
dft_og = DFT(dft_transform,inverse=True)
fft_og = ALL_BASE_FFT(fft_transform,inverse=True)

with open('og.csv','w') as file:
    for i in range(BIG):
        file.write(f'{samples[i]},{dft_og[i]},{fft_og[i]} ,{numpy_og[i]}\n')


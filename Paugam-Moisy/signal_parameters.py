import random
# rand = [i for i in [random.sample(range(0, 10), 3)][0]]

def parameters(signal, ex):

    # задать на какие нейроны приходит сигнал
    n = {'1': [0,1,2,3], '2': [0,3,4,8], '3': [1,3,5,7], '4': [0,1,2,4], '5': [0,1,2,5]}

    # задать в какие моменты спайки приходят на нейроны (в диапазоне от 1 до 20)
    B = [10]
    D = [5,16]
    C = [2,9,17]
    A = [2,7,12,17]

    # A = [2,9,17]
    # B = [3,10,18]
    # C = [4,11,19]
    # D = [5,12,20]

    # A = [2,7,12,17]
    # B = [3,8,13,18]
    # C = [4,9,14,19]
    # D = [5,10,15,20]

    # A = [5,16]
    # B = [6,17]
    # C = [7,18]
    # D = [8,19]

    vector = ''
    n = n[str(ex)]
    A,B,C,D = list(A),list(B),list(C),list(D)
    aA,bB,cC,dD = 0, 0, 0, 0
    
    for s in signal:
        if s == 'A' and aA == 0: 
            aA = 1
            vector += f'A = {A} '
        if s == 'B' and bB == 0:  
            bB = 1
            vector += f'B = {B} '
        if s == 'C' and cC == 0: 
            cC = 1
            vector += f'C = {C} '
        if s == 'D'and dD == 0: 
            dD = 1
            vector += f'D = {D} '

    nA, nB, nC, nD = [],[],[],[]
    for i in range(len(signal)):
        if signal[i] == 'A': nA += [n[i]]
        if signal[i] == 'B': nB += [n[i]]
        if signal[i] == 'C': nC += [n[i]]
        if signal[i] == 'D': nD += [n[i]]
    if nA == []: nA = [9]
    if nB == []: nB = [9]
    if nC == []: nC = [9]
    if nD == []: nD = [9] 

    ABCD_ = ['' for i in range(10)]
    for i in range(10):
        if i in nA and 'A' in signal: ABCD_[i] += 'A'
        if i in nB and 'B' in signal: ABCD_[i] += 'B'
        if i in nC and 'C' in signal: ABCD_[i] += 'C'
        if i in nD and 'D' in signal: ABCD_[i] += 'D'
    keys = [i for i in range(10)]
    labelsAB = dict(zip(keys, ABCD_))

    for i in range(len(A)): A[i] = A[i]/20
    for i in range(len(B)): B[i] = B[i]/20
    for i in range(len(C)): C[i] = C[i]/20
    for i in range(len(D)): D[i] = D[i]/20

    x = [[100] for i in range(4)]
    for i in range(len(signal)):
        if signal[i] == 'A': x[0] = list(A)
        if signal[i] == 'B': x[1] = list(B)
        if signal[i] == 'C': x[2] = list(C)
        if signal[i] == 'D': x[3] = list(D)

    l = [len(i) for i in x]
    for i in x:
        while len(i) < max(l):
            i += [100]

    return labelsAB, vector, nA, nB, nC, nD, n, x
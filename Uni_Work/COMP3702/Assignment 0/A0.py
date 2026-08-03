# -*- coding: utf-8 -*-
"""
Assignment 0 template

For submission, rename this file to "A0.py" 

Answer each question in the corresponding method definition stub below
"""


def Q1(A,B):
    union = set()
    intersection = set()

    # Intersection:
    for element in A:
        if element in B:
            intersection.add(element)
    # Union:
    for element in A:
        union.add(element)

    for element in B:
        union.add(element)

    return union,intersection


def Q2(A,B):
    # Intersection:
    intersection = set()
    for element in A:
        if element in B:
            intersection.add(element)
    
    if intersection == set():
        return 'DISJOINT'
    else:
        return 'INTERSECTING'

def Q3(a,b):
    # Set of Natural Numbers including 0, x < a:
    X = set()
    # Set of Natural Numbers including 0, y < b:
    Y = set()
    # Cartesian Product of X and Y, intersection of x & y:
    G = set()

    # Set X:
    for i in range(a):
        X.add(i)

    # Set Y:
    for i in range(b):
        Y.add(i)

    # Set G:
    for x in X:
        for y in Y:
            G.add((x, y))
            
    return X,Y,G

E = {'e1': (1, 3),
'e2': (2, 3, {'weight': 3.1415}),
'e3': (2, 4),
'e4': (3 ,4)
}


def Q4(E,n):
    n_successors = set()

    keys = list(E.keys())
    key = keys[n]

    for i in range(len(keys)):
        if i == n:
            if E[key][1] == n:
                break
        else:
            n_successors.add(E[key][1])

    return n_successors


def Q5(inFile,outFile,remove):
    with open(inFile, 'r') as infile:
        content = infile.read()
    
    modified_content = content.replace(remove, '')

    with open(outFile, 'w') as outfile:
        outfile.write(modified_content)

    print('Character '+remove+' removed from '+inFile)
    print('Output written to '+outFile)


def Q6(state1,state2):
    #Q6('12345678_','12345_786' returns U.

    #Start with tests:
    # Input lengths:
    if (len(state1) != 9) or (len(state2) != 9):
        print('IMPOSSIBLE')
    
    # Contains Blank:
    if (state1.count('_') != 1) or (state2.count('_') != 1):
        print('IMPOSSIBLE')
    
    startBlank = state1.index('_')
    endBlank = state2.index('_')

    if startBlank == endBlank:
        print('IMPOSSIBLE')
    
    

    print('IMPOSSIBLE')

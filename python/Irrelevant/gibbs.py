import numpy as np
import scipy
import itertools
import pandas as pd
from scipy.stats import binom
from recordlinkage import datasets
import time
import sys


def make_Z_init(n1,n2):
    Z = [n1 + i for i in range(n2)]
    matchX1 = np.random.choice([i for i in range(n1)], size=int(.5*n2), replace=False, p=None)
    matchX2 = np.random.choice([i for i in range(n2)], size=int(.5*n2), replace=False, p=None)
    for i,x2 in enumerate(matchX2):
        Z[x2] = matchX1[i]
    return Z

# function for making prior
def make_prior(init, hypers,L):
    assert len(hypers) == 6, 'invalid hyper'
    init = {"pM": 0.5,
        "pML": [0.5]*L,
        "pUL": [0.5]*L}

## specify hyper parameters
    hypers = {"aM": hypers[0],
            "bM": hypers[1],
            "aML": [1]*hypers[2],
            "bML": [1]*hypers[3],
            "aUL": [1]*hypers[4],
            "bUL": [1]*hypers[5]}
    return init, hypers

# check valid prior
def check_valid_prior(init):
    assert (init["pM"] < 1) and (init['pM'] > 0), 'pM must be between 0,1'
    assert (0 not in init['pML']) and (1 not in init['pML']), 'elements of pML must be b/w 0,1'
    assert (0 not in init['pUL']) and (1 not in init['pUL']), 'elements of pML must be b/w 0,1'
    return

def make_file_names(nPair, L):
    zName = 'Z_trace_nPair' + str(n1*n2) + '_cat' + str(L)
    traceName = 'trace_nPair' + str(n1*n2) + '_cat' + str(L)
    return (zName, traceName)

def sample_pM(s, aM, bM):
    aNew = aM + s.nM
    bNew = bM + n2 - s.nM
    if ((aNew <= 0) or (bNew <=0)): print('neg. beta param')
    return np.random.beta(aNew,bNew)

# updates for pML
def sample_pML(s, aML, bML):
    assert n2 == len(s.Z), 'Z got messed up'
    ones = np.array([1] * L)
    aSums = np.zeros(L)
    bSums = np.zeros(L)
    for x2 in s.matchedX2:
        matchInd = Gamma.index[(Gamma['i']==x2)&(Gamma['j']==s.Z[x2])].tolist()
        for y in matchInd:
            aSums += Gamma.loc[y]['gamma']
            bSums += (ones-Gamma.loc[y]['gamma'])
    aNew = aML + aSums
    bNew = bML + bSums
    if ((any(a <= 0 for a in aNew)) or (any(b <= 0 for b in bNew))): print('neg. beta param')
    return np.random.beta(aNew, bNew)

# updates for pUL
def sample_pUL(s, aUL, bUL):
    assert n2 == len(s.Z), 'Z got messed up'
    ones = np.array([1] * L)
    aSums = np.zeros(L)
    bSums = np.zeros(L)
    for x2 in range(n2):
        nonMatchInd = Gamma.index[(Gamma['i']==x2)&(Gamma['j']!=s.Z[x2])].tolist()
        for y in nonMatchInd:
            aSums += Gamma.loc[y]['gamma']
            bSums += (ones-Gamma.loc[y]['gamma'])
    aNew = aUL + aSums
    bNew = bUL + bSums
    if ((any(a <= 0 for a in aNew)) or (any(b <= 0 for b in bNew))): print('neg. beta param')
    return np.random.beta(aNew, bNew)

#updates for Z
# Move 1 - delete a match
def move_1(s, pM, pML, pUL):

    assert len(s.matchedX2) == s.nM, 'nM is not calibrated correctly'
    assert len(s.matchedX2) == len(s.matchedX1), 'not bipartite matching'

    old_state = State.make_state(s.Z, pM, pML, pUL) # save current Z
    Z_new = old_state.Z

    if (len(old_state.matchedX2) == 0):  # no matches to remove, try another move
        return old_state

    #option 1 - randomly select i in matchedX2 and set to non-match

    i = old_state.matchedX2[np.random.choice(old_state.nM)] # randomly select i in X2
    Z_new[i] = i + n1                          # set i's status to non-match in Z
    new_state = State.make_state(Z_new,pM, pML, pUL)              # make proposal state

    # calculate jump probability pMH
    const = np.log(old_state.nM/((n1-old_state.nM+1)*(n2-old_state.nM+1)))

    pMH = min(1, np.exp(new_state.llh+const-old_state.llh))
#     print('prob of jump is ' + str(pMH))

    accept = np.random.binomial(1, pMH)     # choose jump or not

    if accept == 1:
        return new_state
    else:
        return old_state

# Move 2 - add a match
def move_2(s, pM, pML, pUL):

    assert len(s.matchedX2) == s.nM, 'nM is not calibrated correctly'
    assert len(s.matchedX2) == len(s.matchedX1), 'not bipartite matching'

    old_state = State.make_state(s.Z, pM, pML, pUL) # save current Z, fix any bugs
    Z_new = old_state.Z

    if (len(old_state.unmatchedX2) == 0) or (len(old_state.unmatchedX1) == 0): # nothing to match
        return old_state

    # option 1 - randomly select which record pair to add

    addX2 = old_state.unmatchedX2[np.random.choice(len(old_state.unmatchedX2))] # randomly pick record to give match
    addX1 = old_state.unmatchedX1[np.random.choice(len(old_state.unmatchedX1))] # randomly pick its match

    Z_new[addX2] = addX1                                         # assign new match
    new_state = State.make_state(Z_new, pM, pML, pUL)                                # make proposal state

    # calculate probability of jump
    const = np.log((n1-old_state.nM)*(n2-old_state.nM)/(old_state.nM+1))

    pMH = min(1, np.exp(new_state.llh+const-old_state.llh))
#     print('prob of jump is ' + str(pMH))

    accept = np.random.binomial(1, pMH)

    if accept == 1:
         return new_state
    else:
        return old_state

# Move 3.1 - switch two pairs
def move_3_v1(s, pM, pML, pUL):

    assert len(s.matchedX2) == s.nM, 'nM is not calibrated correctly'
    assert len(s.matchedX2) == len(s.matchedX1), 'not bipartite matching'

    old_state = State.make_state(s.Z, pM, pML, pUL) # save current Z, fix any bugs

    if old_state.nM < 2:
        # nothing to switch
        return old_state

    #Randomly select 2 matched pairs with prob 2/nm(nm-1)
    (i,k) = np.random.choice(old_state.nM, size=2, replace=False, p=None)
    j = old_state.Z[old_state.matchedX2[i]]
    l = old_state.Z[old_state.matchedX2[k]]

    # calculate jump probability pMH
    pMH = calc_pMH_move3(i,j,k,l, pML, pUL)

#     print('prob of jump is ' + str(pMH))
    accept = np.random.binomial(1, pMH)

    if accept == 1:
        # flip entries in Z
        old_state.Z[old_state.matchedX2[i]] = l
        old_state.Z[old_state.matchedX2[k]] = j
        new_state = State.make_state(old_state.Z, pM, pML, pUL)
        return new_state

    else:
        return old_state

def calc_pMH_move3(i, j, k, l, pML, pUL):
    gamma_il = Gamma[(Gamma['i']==int(i)) & (Gamma['j']==int(l))]['gamma'].values[0]
    gamma_kj = Gamma[(Gamma['i']==int(k)) & (Gamma['j']==int(j))]['gamma'].values[0]
    gamma_ij = Gamma[(Gamma['i']==i) & (Gamma['j']==j)]['gamma'].values[0]
    gamma_kl = Gamma[(Gamma['i']==k) & (Gamma['j']==l)]['gamma'].values[0]
    num = calc_pGammaM(gamma_il, pML)+calc_pGammaM(gamma_kj, pML) +\
            calc_pGammaU(gamma_ij, pUL)+calc_pGammaU(gamma_kl, pUL)
    denom = calc_pGammaM(gamma_ij, pML)+calc_pGammaM(gamma_kl, pML) +\
            calc_pGammaU(gamma_il, pUL)+calc_pGammaU(gamma_kj, pUL)
    return min(1, np.exp(num-denom))

# Move 3.2 - Replace one of matching records with nonmatching record
def move_3_v2(s, pM, pML, pUL):

    old_state = State.make_state(s.Z, pM, pML, pUL)

    assert len(old_state.matchedX2) == old_state.nM, 'nM is not calibrated correctly'
    assert len(old_state.matchedX2) == len(old_state.matchedX1), 'not bipartite matching'

    if (len(old_state.unmatchedX1) == 0) and (len(old_state.unmatchedX2)==0):  # impossible to switch
        return old_state

    if (old_state.nM == 0):
        return old_state # no pairs to replace

    if (len(old_state.unmatchedX1) == 0):  # not possible to switch x2's partner
            file == 1

    elif (len(old_state.unmatchedX2)==0):  # not possible to switch x1's partner
            file ==2

    else:  # randomly select x1 or x2 to switch its partner
        file = np.random.randint(2) + 1

    # Randomly select matchedX2, unmatchedX1, and match them

    if file == 1:  # randomly select a file from x1 and replace its partner
        j = old_state.matchedX1[np.random.randint(old_state.nM)]
        l = old_state.unmatchedX2[np.random.randint(len(old_state.unmatchedX2))]
        i = [i for i, val in enumerate(old_state.Z) if val == j][0]

    elif file == 2: # randomly select a file from x2 and replace its partner
        i = old_state.matchedX2[np.random.randint(old_state.nM)]
        l = old_state.unmatchedX1[np.random.randint(len(old_state.unmatchedX1))]
        j = old_state.Z[i]

    # calculate jump probability pMH
    pMH = calc_pMH_move3_v2(i,j,l, pML, pUL)

#     print('prob of jump is ' + str(pMH))
    accept = np.random.binomial(1, pMH)

    if accept == 1:
        Z_new = old_state.Z
        if file == 1: # fix X1's partner
            Z_new[i] = n1 + i # unmatch old X2 partner
            Z_new[l] = j      # add new partner
        elif file == 2: # update X2's partner
            Z_new[i] = l

        new_state = State.make_state(Z_new, pM, pML, pUL)
        return new_state

    else:
        return old_state

def calc_pMH_move3_v2(i,j,l, pML, pUL):
    gamma_il = Gamma[(Gamma['i']==int(i)) & (Gamma['j']==int(l))]['gamma'].values[0]
    gamma_ij = Gamma[(Gamma['i']==int(i)) & (Gamma['j']==int(j))]['gamma'].values[0]
    num = calc_pGammaM(gamma_il, pML)+calc_pGammaU(gamma_ij, pUL)
    denom = calc_pGammaM(gamma_ij, pML)+calc_pGammaU(gamma_il, pUL)
    return min(1, np.exp(num-denom))

# Move 3.3 - delete 1 pair, replace it with new pair
def move_3_v3(s, pM, pML, pUL):

    old_state = State.make_state(s.Z, pM, pML, pUL) # save current Z

    assert len(old_state.matchedX2) == old_state.nM, 'nM is not calibrated correctly'
    assert len(old_state.matchedX2) == len(old_state.matchedX1), 'not bipartite matching'

    if (len(old_state.matchedX2) == 0) or (len(old_state.unmatchedX2) == 0):  # no matches to remove, try another move
        return old_state

    if (len(old_state.matchedX1) == 0) or (len(old_state.unmatchedX1) == 0):  # no matches to remove, try another move
        return old_state

    #randomly select i in matchedX2 and set to non-match

    i = old_state.matchedX2[np.random.choice(old_state.nM)] # randomly select i in X2
    unmatchedX1 = old_state.unmatchedX1 + [old_state.Z[i]]  # free old match
    newZ = old_state.Z
    newZ[i] = i + n1                          # set i's status to non-match in Z

    # randomly select j in unmatched X2 and set to match
    addX2 = old_state.unmatchedX2[np.random.choice(len(old_state.unmatchedX2))]
    j = unmatchedX1[np.random.choice(len(unmatchedX1))]
    newZ[addX2] = j

    new_state = State.make_state(newZ, pM, pML, pUL)              # make proposal state

    # calculate jump probability
    const = np.log(old_state.nM/((n1-old_state.nM)*(n2-old_state.nM)))
    pMH = min(1, np.exp(new_state.llh + const - old_state.llh))

#     print('prob of jump is ' + str(pMH))

    # choose jump or not
    accept = np.random.binomial(1, pMH)

    if accept == 1:
        return new_state

    else:
        return old_state

class State(object):
    matchedX2 = []
    unmatchedX2 = []
    matchedX1 = []
    unmatchedX1 = []
    Z = []
    nM = 0
    llh = 1

    def __init__(self,Z):
        self.matchedX2 = [i for i in range(n2) if Z[i] < n1]
        self.matchedX1 = [i for i in Z if i < n1]
        self.nM = len(self.matchedX2)
        self.unmatchedX2 = np.delete([i for i in range(n2)], self.matchedX2)
        self.unmatchedX1 = [i for i in range(n1) if i not in Z]
        self.Z = Z

        if len(Z) != len(np.unique(Z)):
            print_state(self)

        assert len(Z) == n2, 'invalid bpm'
        assert len(Z) == len(np.unique(Z)), 'invalid bpm'

    def make_state(Z, pM, pML, pUL):
        state = State(Z)
        state.llh = calc_pNM_Z(state, pM, pML, pUL)
        return state

    def print_state(s):
        print('Z: ' + str(s.Z))
        print('matchedX2: ' + str(s.matchedX2))
        print('matchedX1: ' + str(s.matchedX1))
        print('unmatchedX2: ' + str(s.unmatchedX2))
        print('unmatchedX1: ' + str(s.unmatchedX1))
        return

# Functions for evaluating log llh of Z
# BDA3 says jump probabilities should be exp(diff in log density)

def calc_pGammaM(gammaInd,pML):
    # returns log(pGamma_ij | M)
    assert len(gammaInd) == len(pML), 'dim do not match'
    return np.sum([gammaInd[l]*np.log(pML[l]) + (1-gammaInd[l])*np.log(1-pML[l]) for l in range(len(pML))])

def calc_pGammaU(gammaInd,pUL):
    # returns log(pGamma_ij | U)
    assert len(gammaInd) == len(pUL), 'dim do not match'
    return np.prod([(pUL[l]**gammaInd[l])*(1-pUL[l])**(1-gammaInd[l]) for l in range(len(pML))])

def calc_pGamma(s, pM, pML, pUL):
    # calculates log P(gamma | Z, pML, pUL) for ENTIRE gamma (which has n1*n2 entries)
    pGamma = 0
    for index, row in Gamma.iterrows():
        if row['j'] == s.Z[row['i']]:
            pGamma += calc_pGammaM(row['gamma'],pML)
        else:
            pGamma += calc_pGammaU(row['gamma'],pUL)
    return pGamma

def calc_pNM_Z(s, pM, pML, pUL):
    #returns log P(nM, Z | current params) ~ log P(nM | pM) + log P(Z | nM) + log P(gamma | all param, Z)
    pNM = np.log(binom.pmf(s.nM, n2, pM))  # p(nM | pM) ~ Binom(nM successes out of n2, w/ param pM)
    pZ = np.log(scipy.math.factorial(n1-s.nM)/scipy.math.factorial(n1))
    pGamma = calc_pGamma(s, pM, pML, pUL)

    return pNM + pZ + pGamma

def gibbs(Gamma, iters, init, hypers, Z_init, n1, n2):
    # some assertions needed here
    assert len(init["pML"]) == len(init["pUL"]), 'dimensions of pML and pUL do not match'
    pM = init["pM"]
    pML = init["pML"]
    pUL = init["pUL"]
    nPar = 1 + len(pML) + len(pUL)

    L = len(pML)
    numPair = Gamma.shape[0]

    trace = np.zeros((iters,nPar+1))
    Z_trace = np.zeros((iters,n2))

    #initialize global state vars
    state = State.make_state(Z_init, pM, pML, pUL)

    for it in range(iters):

        # 1. Draw pM | Z ~ Beta(aM + nM(Z), bM + n2 - nM(Z))
        pM = sample_pM(state, hypers['aM'], hypers['bM'])

        # 2. Draw pML | Z, pUL | Z
        gamma = Gamma['gamma']
        pML = sample_pML(state, hypers['aML'], hypers['bML'])
        pUL = sample_pUL(state, hypers['aUL'], hypers['bUL'])

        # 3. Draw values of Z (update state)
        moveType = np.random.randint(3) + 1
        if moveType == 3:
            m3 = np.random.randint(3) + 1
            moveType = '3_v' + str(m3)
        move = 'move_' + str(moveType) + '(state, pM, pML, pUL)'
#         print('move' + str(moveType))
        state = eval(move)

        trace[it,:] = np.append(np.append(np.append(pM, pML),pUL),state.llh) # update trace
        Z_trace[it, :] = state.Z

    trace = pd.DataFrame(trace)
    Z_trace = pd.DataFrame(Z_trace)
    Z_trace = Z_trace.astype(int)

    pML_names = ['pML_' + str(i) for i in range(1,L+1)]
    pUL_names = ['pUL_' + str(i) for i in range(1,L+1)]
    trace.columns= ['pM'] + pML_names + pUL_names + ['llh']
    return trace, Z_trace

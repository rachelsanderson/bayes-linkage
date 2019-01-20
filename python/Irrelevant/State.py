import State
import numpy as np
import scipy
import itertools
import pandas as pd
from scipy.stats import binom
n1 = 10
n2 = 10

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

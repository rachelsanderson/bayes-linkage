from recordlinkage import datasets
import numpy as np
import pandas as pd

def make_fake_data(n1, n2, pM, pML, pUL, randState = 113):
    nPair = n1 * n2
    L = len(pML)
    gamma, links =np.array(datasets.binary_vectors(nPair, int(pM*nPair), \
                m=pML, u = pUL, random_state=randState, return_links = True))

    gamma['match'] = False
    gamma.loc[links,'match']= True
    matches = gamma['match']
    # make pair identifiers
    i = [[i]*n1 for i in range(n2)]
    iVals = []
    for x in i:
        iVals += x
    jVals = [j for j in range(n1)] * n2

    Gamma = pd.DataFrame(
        {'gamma': list(gamma[['c_1','c_2','c_3']].values),
        'i': iVals,
        'j': jVals,
        'match': matches})
    Gamma = Gamma.reset_index(drop=True)
    ext = 'nMatch' + str(int(pM*nPair)) + '_L' + str(L)
    Gamma.to_csv('Gamma_'+ext+'.csv', mode='w')
    return Gamma

if __name__ == "__main__":
    n1 = n2 = 10
    pML = [.8] * 4
    pUL = [.2] * 4
    pM = .2
    make_fake_data(n1, n2, pM, pML, pUL)

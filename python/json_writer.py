import json
import sys

def write_params(L,n1,n2,pM,pML,pUL):
    params = {}
    params['L'] = L
    params['init'] = {"pM": 0.5,
                      "pML": [0.5]*L,
                       "pUL": [0.5]*L}
    params['hypers'] = {"aM": 1,
                "bM": 1,
                "aML": [1]*L,
                "bML": [1]*L,
                "aUL": [1]*L,
                "bUL": [1]*L}
    params['n1'] = n1
    params['n2'] = n2
    params['pM'] = pM
    params['pML'] = [pML] * L
    params['pUL'] = [pUL] * L

    fileName = 'param_nMatch' + str(int(pM*n1*n2)) + '_L' + str(L) + '.json'

    with open(fileName,'w') as outfile:
        json.dump(params,outfile)
    return fileName

if __name__ == '__main__':
    L  = int(sys.argv[1])
    n1 = int(sys.argv[2])
    n2 = int(sys.argv[3])
    pM = float(sys.argv[4])
    pML = float(sys.argv[5])
    pUL = float(sys.argv[6])
    write_params(L,n1,n2,pM,pML,pUL)

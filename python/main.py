import numpy as np
import pandas as pd
import gibbs
import make_fake_data as make_data
import json
import time
import sys

if __name__ == "__main__":

    # ARGUMENTS
    L=4
    init = {"pM": 0.5,
        "pML": [0.5]*L,
        "pUL": [0.5]*L}

    hypers = {"aM": 1,
              "bM": 1,
              "aML": [1]*L,
              "bML": [1]*L,
              "aUL": [1]*L,
              "bUL": [1]*L}
    n1 = n2 = 10
    pML = [.8] * 4
    pUL = [.2] * 4
    pM = .2
    niters = 2
    # READ IN ARGUMENTS
    # hypers_file = int(sys.argv[1])
    # hypers = json.loads(hypers_file)

    # CHECK VALID INPUT

    # (init_dict, hypers_dict) = make_prior(init,hypers,L)
    # check_valid_prior(init)

    # MAKE FAKE DATA WITH ARGUMENTS
    Gamma = make_data.make_fake_data(n1,n2,pM,pML,pUL)

    # MAKE INITIAL Z
    Z_init = gibbs.make_Z_init(n1, n2)

    # perform gibbs on fake data
    start = time.time()
    trace, Z_trace = gibbs.gibbs(Gamma, niters, init, hypers, Z_init, n1, n2)
    end = time.time()
    print(end-start)
    zName, traceName = make_file_names(Gamma.shape[0], L)
    Z_trace.to_csv(zName+'.csv', mode='w')
    trace.to_csv(traceName+'.csv', mode='w')

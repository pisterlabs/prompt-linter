import time
from output import write_coherence_matrices
from coherence import get_coherence_matrices, get_synchronous_coherence_matrices
import numpy as np
import os
import json

# loadmat
from scipy.io import loadmat


def test_coherence_methods():
    file_name = (
        r"C:\Users\saarm\Code Projects\My-CoG\PyCoG\users_data\bp_fingerflex.mat"
    )
    data = loadmat(file_name)["data"]
    # measure the time it takes to compute the coherence matrices
    time_async = time.time()
    f_async, CM_async = get_coherence_matrices(data, 1000, "hann", 0.5)
    end_time_async = time.time()

    time_synch = time.time()
    f_synch, CM_synch = get_synchronous_coherence_matrices(data, 1000, "hann", 0.5)
    end_time_synch = time.time()

    write_coherence_matrices(f_synch, CM_synch, "synch.json")
    write_coherence_matrices(f_async, CM_async, "async.json")

    print(f"synch time: {end_time_synch - time_synch}")
    print(f"async time: {end_time_async - time_async}")
    return


print("yalla")
test_coherence_methods()

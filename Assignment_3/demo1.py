import numpy as np
from scipy.io import loadmat
from spectral_clustering import spectral_clustering

def main():
    # # Φορτωση προϋπολογισμενου affinity πίνακα από τα δεδομενα
    data = loadmat("dip_hw_3.mat")
    d1a = data["d1a"]

    # Εκτελεση spectral clustering για k = 2, 3 και 4 clusters
    for k in [2, 3, 4]:
        labels = spectral_clustering(d1a, k)     # Spectral clustering για τον W
        print(f"Spectral clustering for k = {k}:")
        print(labels)
        print()

if __name__ == "__main__":
    main()
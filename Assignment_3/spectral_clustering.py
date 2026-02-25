import numpy as np
from scipy.sparse.linalg import eigs  # Για εύρεση ιδιοδιανυσμάτων αραιών πινάκων
from sklearn.cluster import KMeans    # Clustering αλγόριθμος (εφαρμόζεται στο φάσμα του γράφου)

def spectral_clustering(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    """
    Εκτελεί spectral clustering με βάση τον affinity πίνακα.

    Παραμετροι:
        affinity_mat: (MN, MN) numpy array, πλήρης affinity matrix
        k: αριθμός των clusters

    Επιστρεφει:
        cluster_idx: (MN,) numpy array dtype=float, με τις ετικέτες των clusters
    """
    # 1. Laplacian matrix: L = D - W
    D = np.diag(np.sum(affinity_mat, axis=1))  # Υπολογίζουμε τον βαθμό (degree) κάθε κόμβου
    L = D - affinity_mat  # Laplacian πίνακας: αποτυπώνει τη δομή του γράφου

    # 2. k ιδιοδιανυσματα που αντιστοιχουν στις k μικροτερες ιδιοτιμες
    eigvals, eigvecs = eigs(L, k=k, which='SM')  # Παίρνουμε τα k ιδιοδιανύσματα
    U = np.real(eigvecs)  # Κρατάμε μόνο το πραγματικό μέρος, γιατί ο πίνακας είναι συμμετρικός

    # 3. KMeans clustering στις γραμμες του U
    kmeans = KMeans(n_clusters=k, random_state=1)  # Ορισμός KMeans
    kmeans.fit(U)  # Εφαρμογη k means στα U
    labels = kmeans.labels_  # Οι ετικέτες (σε ποιο cluster ανήκει κάθε pixel)

    # 4. Μετατροπη σε float
    cluster_idx = labels.astype(float)  

    return cluster_idx  # Επιστροφή ετικετών των pixels


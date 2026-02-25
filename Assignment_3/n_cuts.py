import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans

def n_cuts(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    """
    Μη-αναδρομική εκδοχή της normalized cuts.
    Διαχωρίζει τον γράφο (pixels) σε k clusters με βάση το φασματικό του αποτύπωμα.

    Παραμετροι:
        affinity_mat: (MN, MN) affinity matrix
        k: αριθμός των clusters

    Επιστρεφει:
        cluster_idx: (MN,) array τύπου float με τα labels των κόμβων
    """
    # 1. Υπολογισμος Λαπλασιανου πινακα L = D - W
    D = np.diag(np.sum(affinity_mat, axis=1))   # Πινακας βαθμων
    L = D - affinity_mat   # Laplacian

    # 2. Επιλυση προβληματος ιδιοτιμων Lx = λDx 
    eigvals, eigvecs = eigs(L, M=D, k=k, which='SM')     # k ιδιοδιανύσματα με τις μικρότερες ιδιοτιμές
    U = np.real(eigvecs)   # Κρατάμε μόνο το πραγματικό μέρος

    # 3. 3. Clustering με KMeans πάνω στα ιδιοδιανύσματα (φασματικός χώρος)
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(U)       # Εφαρμογη k means στα U
    labels = kmeans.labels_       # Ποιο pixel ανήκει σε ποιο cluster

    # 4. Επιστροφη ετικετων σε float μορφη
    cluster_idx = labels.astype(float)
    
    return cluster_idx



def calculate_n_cut_value(affinity_mat: np.ndarray, cluster_idx: np.ndarray) -> float:
    """
    Υπολογίζει την τιμή της μετρικής Ncut μεταξύ των δύο clusters.

    Προϋπόθεση: ο διαχωρισμός πρέπει να έχει ακριβώς 2 ομάδες (labels 0 και 1).

    Παραμετροι:
        affinity_mat: (MN, MN) affinity matrix
        cluster_idx: (MN,) labels με 2 μοναδικές τιμές (π.χ. 0 και 1)

    Επιστρεφει:
        n_cut_value: float
    """
    assert affinity_mat.shape[0] == len(cluster_idx)    # Εξασφαλίζουμε σωστή διάσταση
    
    # Βρίσκουμε τα indices των κόμβων ανά cluster
    A = np.where(cluster_idx == 0)[0]
    B = np.where(cluster_idx == 1)[0]

    # Ομοιογένεια εντός των clusters (assoc)
    assoc_AA = np.sum(affinity_mat[np.ix_(A, A)])
    assoc_AV = np.sum(affinity_mat[A, :])

    assoc_BB = np.sum(affinity_mat[np.ix_(B, B)])
    assoc_BV = np.sum(affinity_mat[B, :])

    # Υπολογισμός normalized association
    n_assoc = (assoc_AA / assoc_AV) + (assoc_BB / assoc_BV)

    # Η τιμή του Ncut = 2 - normalized association
    n_cut_value = 2.0 - n_assoc

    return n_cut_value


def n_cuts_recursive(affinity_mat: np.ndarray, T1: int, T2: float, depth: int = 0) -> np.ndarray:
    """
    Αναδρομική εκδοχή του normalized cuts: συνεχίζει να διαχωρίζει το γράφο
    όσο ικανοποιούνται τα κατώφλια T1 και T2.
    
    Παραμετροι:
        affinity_mat: (MN, MN) affinity matrix
        T1: ελάχιστο μέγεθος cluster για να διαχωριστεί, αν πεσει κατω απο αυτο σταματαει
        T2: μέγιστη αποδεκτή τιμή Ncut για να συνεχιστεί η διαίρεση, αν το ξεπερασει σταματαει

    Επιστρεφει:
        cluster_idx: (MN,) πίνακας float labels για κάθε κόμβο
    """
    indent = "  " * depth  # Για να φαίνεται το βάθος αναδρομής στην εκτύπωση
    cluster_idx = n_cuts(affinity_mat, k=2)

    A = np.where(cluster_idx == 0)[0]
    B = np.where(cluster_idx == 1)[0]

    n_cut_val = calculate_n_cut_value(affinity_mat, cluster_idx)
    print(f"{indent}Examining cluster of size {len(cluster_idx)}: Ncut = {n_cut_val:.4f}")

    if len(A) < T1 or len(B) < T1:
        print(f"{indent}Stopping (T1). Subcluster too small: |A| = {len(A)}, |B| = {len(B)}")
        return cluster_idx
    
    if n_cut_val > T2:
        print(f"{indent}Stopping (T2). Ncut too high: {n_cut_val:.4f} > {T2}")
        return cluster_idx
    
    print(f"{indent}Splitting cluster into two parts")

    labels_A = n_cuts_recursive(affinity_mat[np.ix_(A, A)], T1, T2)
    labels_B = n_cuts_recursive(affinity_mat[np.ix_(B, B)], T1, T2)

    cluster_idx[A] = labels_A
    cluster_idx[B] = labels_B + np.max(labels_A) + 1

    return cluster_idx


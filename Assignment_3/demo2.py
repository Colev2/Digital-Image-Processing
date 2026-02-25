import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering

# Φόρτωση εικόνων 
data = loadmat("dip_hw_3.mat")
d2a = data["d2a"]
d2b = data["d2b"]

# Αποθηκευση
images = {"d2a": d2a, "d2b": d2b}

# Για κάθε εικόνα
for name, img in images.items():
    print(f"\nProcessing image: {name}")

    # Δημιουργία affinity matrix
    affinity_mat = image_to_graph(img)

    # Για κάθε τιμή του k
    for k in [2, 3, 4]:
        print(f"Running spectral clustering for k = {k}...")

        # Εκτέλεση clustering
        cluster_idx = spectral_clustering(affinity_mat, k)

        # Πλήθος στοιχείων σε κάθε cluster
        counts = np.unique(cluster_idx, return_counts=True)[1]
        print(f"Counts per cluster: {counts}")

        # Αναδιαμορφωση αποτελεσματος σε εικονα (2D)
        H, W, _ = img.shape
        clustered_img = cluster_idx.reshape(H, W)

        # Εμφάνιση εικόνας με ψευδο-χρωματισμό ανά cluster
        plt.figure()
        plt.imshow(clustered_img)
        plt.title(f"{name} - Spectral Clustering (k={k})")
        plt.axis("off")
            
plt.show()


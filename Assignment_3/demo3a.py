import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from image_to_graph import image_to_graph
from n_cuts import n_cuts


# Φορτωση εικονων
data = loadmat("dip_hw_3.mat")
d2a = data["d2a"]
d2b = data["d2b"]

# Αποθηκευση
images = {"d2a": d2a, "d2b": d2b}

# Για κάθε εικόνα
for name, img in images.items():
    print(f"\nProcessing image: {name}")

    # Μετατροπη εικονας σε γράφο, δηλαδή παίρνουμε τον πίνακα affinity
    affinity_mat = image_to_graph(img)

    # Clustering για κάθε τιμή του k
    for k in [2, 3, 4]:
        print(f"  Running n-cuts clustering for k = {k}...")

        # Εκτελεση normalized cuts για το συγκεκριμένο k
        cluster_idx = n_cuts(affinity_mat, k)

        # Υπολογισμος πληθους pixels που μπήκαν σε κάθε cluster
        counts = np.unique(cluster_idx, return_counts=True)
        print(f"  Counts per cluster: {counts}")

        # Μετατροπη του 1D array των labels σε εικόνα 2D
        H, W, _ = img.shape
        clustered_img = cluster_idx.reshape(H, W)

        # Εμφανιση segmentation
        plt.figure()
        plt.imshow(clustered_img)  
        plt.title(f"{name} - nCuts (k={k})")
        plt.axis("off")  


plt.show()


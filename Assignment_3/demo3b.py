import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from image_to_graph import image_to_graph
from n_cuts import n_cuts
from n_cuts import calculate_n_cut_value
from spectral_clustering import spectral_clustering


# Φόρτωση εικόνων
data = loadmat("dip_hw_3.mat")
d2a = data["d2a"]
d2b = data["d2b"]

for name, img in [("d2a", d2a), ("d2b", d2b)]:
    print(f"\nProcessing image: {name}")

    # Δημιουργία γράφου
    affinity_mat = image_to_graph(img)

    # Εφαρμογή non-recursive n-cuts με k=2
    labels = n_cuts(affinity_mat, k=2)

    # Υπολογισμός Ncut value
    ncut_val = calculate_n_cut_value(affinity_mat, labels)
    print(f"Ncut value: {ncut_val:.4f}")

    # Spectral clustering για σύγκριση
    labels_spec = spectral_clustering(affinity_mat, k=2)

    # Οπτικοποίηση αποτελεσματων σε ενα figure με 3 εικονες 
    plt.figure(figsize=(10, 4))
        
    # Αριστερα: αρχικη εικονα
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    # Μεση: αποτελεσμα απο normalized cuts
    plt.subplot(1, 3, 2)
    plt.imshow(labels.reshape(img.shape[:2]))
    plt.title(f"{name} - nCuts (k=2)\nNcut = {ncut_val:.4f}")
    plt.axis('off')

    # Δεξια: αποτελεσμα απο spectral clustering
    plt.subplot(1, 3, 3)
    plt.imshow(labels_spec.reshape(img.shape[:2]))
    plt.title("Spectral Clustering (k=2)")
    plt.axis('off')

    plt.tight_layout()
    
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from image_to_graph import image_to_graph
from n_cuts import n_cuts, n_cuts_recursive
from spectral_clustering import spectral_clustering

def main():

    # Φορτωση δεδομενων
    data = loadmat("dip_hw_3.mat")
    d2a = data["d2a"]
    d2b = data["d2b"]

    # Ορισμος Threshold
    T1 = 5
    T2 = 0.9

    for name, img in [("d2a", d2a), ("d2b", d2b)]:
        print(f"\nProcessing image: {name}")

        # Μετατροπη εικονας σε γραφο, Δημιουργια affinity
        affinity = image_to_graph(img)

        # Αναδρομικη normalized cuts
        labels_recursive = n_cuts_recursive(affinity, T1, T2)
        
        # Μη-αναδρομικη για k=2 και k=3
        labels_ncuts_k2 = n_cuts(affinity, k=2)
        labels_ncuts_k3 = n_cuts(affinity, k=3)

        # Spectral clustering για k=2 και k=3
        labels_spectral_k2 = spectral_clustering(affinity, k=2)
        labels_spectral_k3 = spectral_clustering(affinity, k=3)

        # Plot figure 2x3
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{name.upper()} - Demo 3c", fontsize=16)

        # 1. Original εικονα
        axs[0, 0].imshow(img)
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")

        # 2. Αναδρομικο ncuts
        axs[0, 1].imshow(labels_recursive.reshape(img.shape[:2]))
        axs[0, 1].set_title(f"Recursive nCuts (T1={T1}, T2={T2})")
        axs[0, 1].axis("off")

        # 3. Μη-αναδρομικο ncuts με k=2
        axs[0, 2].imshow(labels_ncuts_k2.reshape(img.shape[:2]))
        axs[0, 2].set_title("nCuts (k=2)")
        axs[0, 2].axis("off")

        # 4. Μη-αναδρομικο ncuts με k=3
        axs[1, 0].imshow(labels_ncuts_k3.reshape(img.shape[:2]))
        axs[1, 0].set_title("nCuts (k=3)")
        axs[1, 0].axis("off")

        # 5. Spectral clustering με k=2
        axs[1, 1].imshow(labels_spectral_k2.reshape(img.shape[:2]))
        axs[1, 1].set_title("Spectral Clustering (k=2)")
        axs[1, 1].axis("off")

        # 6. Spectral clustering με k=3
        axs[1, 2].imshow(labels_spectral_k3.reshape(img.shape[:2]))
        axs[1, 2].set_title("Spectral Clustering (k=3)")
        axs[1, 2].axis("off")

        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
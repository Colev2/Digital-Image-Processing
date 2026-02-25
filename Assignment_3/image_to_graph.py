import numpy as np
from scipy.spatial.distance import cdist  # Υπολογισμός αποστάσεων μεταξύ σημείων (pixels)

def image_to_graph(img_array: np.ndarray) -> np.ndarray:
    """
    Μετατρέπει την εικόνα σε γράφο: κάθε pixel είναι node, και το βάρος κάθε ακμής
    είναι 1 / exp(ευκλείδεια απόσταση των χρωμάτων τους).

    Παραμετροι:
        img_array: np.ndarray of shape (M, N, C), με τιμές [0, 1] (RGB εικόνα κανονικοποιημένη)
    
    Επιστρεφει:
        affinity_mat: np.ndarray of shape (MN, MN), ο πλήρης πίνακας συγγένειας
    """

    # Διαστάσεις της εικόνας: M = ύψος, N = πλάτος, C = αριθμός καναλιών (συνήθως 3 για RGB)
    M, N, C = img_array.shape
    num_pixels = M * N  # Συνολικός αριθμός pixels

    # Βήμα 1: Κάνουμε flatten την εικόνα — κάθε pixel γίνεται γραμμή (με τα RGB χαρακτηριστικά του)
    flat_img = img_array.reshape((num_pixels, C))  # Σχήμα: (MN, C)

    # Βήμα 2: Υπολογίζουμε τις ευκλείδειες αποστάσεις ανάμεσα σε όλα τα ζεύγη pixels
    # Αποτέλεσμα: ένας τετράγωνος πίνακας αποστάσεων διαστάσεων (MN, MN)
    dists = cdist(flat_img, flat_img, metric='euclidean')

    # Βήμα 3: Υπολογισμός πίνακα συγγένειας βάσει απόστασης
    # Όσο μικρότερη η απόσταση, τόσο μεγαλύτερη η συγγένεια
    # Η συγγένεια ορίζεται ως 1 / exp(απόσταση)
    affinity_mat = 1.0 / np.exp(dists)

    # Επιστρέφουμε τον πλήρη affinity πίνακα
    return affinity_mat
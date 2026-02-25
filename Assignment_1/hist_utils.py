import numpy as np
from typing import Dict

# -------- Αρχή συναρτήσεων --------

def calculate_hist_of_img(img_array: np.ndarray, return_normalized: bool) -> Dict:
    """
    Υπολογίζει το ιστόγραμμα μιας εικόνας.
    Αν return_normalized=True, επιστρέφει κανονικοποιημένο ιστόγραμμα.
    """
    
    hist = {}  # Δημιουργία κενού λεξικού για το ιστόγραμμα

    # Εύρεση όλων των μοναδικών τιμών που υπάρχουν στην εικόνα
    unique_values = np.unique(img_array)

    # Για κάθε μοναδική τιμή, υπολογίζουμε πόσες φορές εμφανίζεται
    for value in unique_values:
        count = np.sum(img_array == value)  # Μετράμε τα pixels που έχουν αυτή την τιμή
        hist[value] = count  # Εισαγωγή της μέτρησης στο λεξικό

    # Αν ζητείται κανονικοποίηση του ιστογράμματος
    if return_normalized:
        total_samples = img_array.size  # Συνολικός αριθμός pixels
        for key in hist:
            hist[key] = hist[key] / total_samples  # Κανονικοποίηση counts σε σχετικές συχνότητες

    return hist  # Επιστροφή του ιστογράμματος

# --------

def apply_hist_modification_transform(img_array: np.ndarray, modification_transform: Dict) -> np.ndarray:
    """
    Εφαρμόζει έναν μετασχηματισμό τροποποίησης ιστογράμματος στην εικόνα.
    Κάθε τιμή εισόδου fi μετατρέπεται στη νέα τιμή εξόδου modification_transform[fi].
    """

    # Δημιουργία ενός κεντρικού πίνακα για την τελική μετασχηματισμένη εικόνα
    modified_img = np.zeros_like(img_array)

    # Για κάθε μοναδική τιμή στην είσοδο:
    # - Βρίσκουμε τα pixels που έχουν αυτή την τιμή
    # - Τα αλλάζουμε σύμφωνα με τον μετασχηματισμό
    for original_value, new_value in modification_transform.items():
        modified_img[img_array == original_value] = new_value

    return modified_img  # Επιστροφή της μετασχηματισμένης εικόνας

# -------- Τέλος --------
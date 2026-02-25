import numpy as np
from typing import Dict
from hist_utils import calculate_hist_of_img, apply_hist_modification_transform

# -------- Αρχή συναρτήσεων --------

def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict, mode: str) -> np.ndarray:
    """
    Μετασχηματίζει το ιστόγραμμα της εικόνας εισόδου ώστε να ταιριάζει στο επιθυμητό ιστόγραμμα αναφοράς,
    σύμφωνα με τον επιλεγμένο αλγόριθμο (mode).
    """

    # Βεβαιωνόμαστε ότι το mode είναι αποδεκτό
    assert mode in ["greedy", "non-greedy", "post-disturbance"], "Invalid mode!"

    # Υπολογισμός ιστογράμματος εισόδου (όχι κανονικοποιημένο)
    input_hist = calculate_hist_of_img(img_array, return_normalized=False)
    total_samples = img_array.size  # Συνολικός αριθμός pixels

    # Επίπεδα φωτεινότητας εισόδου και εξόδου ταξινομημένα
    input_levels = sorted(input_hist.keys())
    output_levels = sorted(hist_ref.keys())

    # Υπολογισμός επιθυμητών αριθμών pixels για κάθε επίπεδο φωτεινότητας στην έξοδο
    desired_counts = {lvl: int(round(hist_ref[lvl] * total_samples)) for lvl in output_levels}

    # Αν χρησιμοποιείται post-disturbance mode
    if mode == "post-disturbance":
        # Υπολογισμός διαφοράς ανάμεσα σε διαδοχικές τιμές
        unique_vals = np.unique(img_array)
        d = unique_vals[1] - unique_vals[0] if len(unique_vals) > 1 else 1.0 / 255.0

        # Εισαγωγή τυχαίου θορύβου ώστε να σπάσουν συσσωρεύσεις τιμών
        noise = np.random.uniform(-d/2, d/2, size=img_array.shape)
        img_array = np.clip(img_array + noise, 0.0, 1.0)  # Φροντίζουμε οι τιμές να παραμένουν εντός [0,1]

        # Στρογγυλοποίηση τιμών για να περιοριστεί ο αριθμός διαφορετικών επιπέδων
        img_array = np.round(img_array, decimals=4)

        # Επανάληψη υπολογισμού ιστογράμματος μετά το θόρυβο
        input_hist = calculate_hist_of_img(img_array, return_normalized=False)
        input_levels = sorted(input_hist.keys())

    # Αρχικοποίηση μεταβλητών για mapping εισόδου-εξόδου
    mapping = {}
    input_idx = 0
    output_idx = 0
    count_accum = 0  # Συγκεντρωτικός μετρητής pixels για κάθε στάθμη

    # Διαδικασία αντιστοίχισης επιπέδων φωτεινότητας
    while input_idx < len(input_levels) and output_idx < len(output_levels):
        current_input = input_levels[input_idx]
        current_output = output_levels[output_idx]

        if mode == "non-greedy":
            # Υπολογισμός υπολειπόμενης ανάγκης pixels για το τρέχον επίπεδο εξόδου
            deficiency = desired_counts[current_output] - count_accum

            if deficiency >= input_hist[current_input] / 2:
                # Αν το έλλειμμα είναι αρκετά μικρό, κρατάμε την αντιστοίχιση
                mapping[current_input] = current_output
                count_accum += input_hist[current_input]
                input_idx += 1
                continue

        # Κανονική αντιστοίχιση εισόδου σε έξοδο
        mapping[current_input] = current_output
        count_accum += input_hist[current_input]

        # Αν καλύψαμε το απαιτούμενο πλήθος για το τρέχον output επίπεδο
        if count_accum >= desired_counts[current_output]:
            output_idx += 1  # Προχωράμε στο επόμενο επίπεδο εξόδου
            count_accum = 0  # Επαναφορά μετρητή

        input_idx += 1

    # Εφαρμογή του τελικού mapping στις τιμές της εικόνας
    modified_img = apply_hist_modification_transform(img_array, mapping)

    return modified_img

# --------

def perform_hist_eq(img_array: np.ndarray, mode: str) -> np.ndarray:
    """
    Εκτελεί εξισορρόπηση ιστογράμματος της εικόνας.
    Δημιουργεί έναν στόχο ιστόγραμμα που κατανέμει ομοιόμορφα όλα τα επίπεδα φωτεινότητας.
    """

    # Υπολογισμός ιστογράμματος εισόδου
    input_hist = calculate_hist_of_img(img_array, return_normalized=False)
    num_levels = len(input_hist)  # Πόσες διαφορετικές τιμές υπάρχουν
    uniform_prob = 1.0 / num_levels  # Στόχος: ίση πιθανότητα σε κάθε επίπεδο

    # Δημιουργία ιστογράμματος-στόχου για ομοιόμορφη κατανομή
    hist_ref = {lvl: uniform_prob for lvl in sorted(input_hist.keys())}

    # Εκτέλεση του βασικού αλγορίθμου μετασχηματισμού
    return perform_hist_modification(img_array, hist_ref, mode)

# --------

def perform_hist_matching(img_array: np.ndarray, img_array_ref: np.ndarray, mode: str) -> np.ndarray:
    """
    Εκτελεί αντιστοίχιση ιστογράμματος: μετασχηματίζει την είσοδο ώστε να μοιάζει με την εικόνα αναφοράς.
    """

    # Υπολογισμός κανονικοποιημένου ιστογράμματος εικόνας αναφοράς
    ref_hist = calculate_hist_of_img(img_array_ref, return_normalized=True)

    # Εκτέλεση μετασχηματισμού ώστε το ιστόγραμμα εισόδου να προσεγγίσει το ιστόγραμμα αναφοράς
    return perform_hist_modification(img_array, ref_hist, mode)

# -------- Τέλος --------

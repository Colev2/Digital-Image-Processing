import numpy as np
import cv2
from fir_conv import fir_conv

def log_edge(in_img_array: np.ndarray) -> np.ndarray:
    """
    Ανίχνευση ακμών με χρήση Laplacian of Gaussian (LoG).
    Επιστρέφει δυαδική εικόνα ακμών μέσω εντοπισμού zero-crossings.
    """

    # Ορισμός LoG μάσκας 5x5 (Laplacian of Gaussian)
    # Περιέχει και εξομάλυνση (Gaussian) και ανίχνευση καμπυλότητας (Laplacian)
    log_mask = np.array([
        [0,  0,  -1,  0,  0],
        [0, -1,  -2, -1,  0],
        [-1, -2,  16, -2, -1],
        [0, -1,  -2, -1,  0],
        [0,  0,  -1,  0,  0]
    ], dtype=float)

    # Θέσεις προέλευσης στην εικόνα και στη μάσκα (κέντρο)
    in_origin = np.array([0, 0], dtype=int)
    mask_origin = np.array([2, 2], dtype=int)

    # Εφαρμογή συνέλιξης της εικόνας με τη LoG μάσκα
    log_result, _ = fir_conv(in_img_array, log_mask, in_origin, mask_origin)

    # Αρχικοποίηση εξόδου (εικόνα ακμών)
    out_img_array = np.zeros_like(log_result, dtype=int)
    rows, cols = log_result.shape

    # Έλεγχος zero-crossings με ενίσχυση φιλτραρίσματος:
    # Μόνο αν υπάρχει αλλαγή πρόσημου (min<0 και max>0) ΚΑΙ
    # Η διαφορά είναι επαρκώς μεγάλη (> 0.5) θεωρείται έγκυρη ακμή
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            patch = log_result[i-1:i+2, j-1:j+2]  # τοπικό παράθυρο 3x3
            if patch.max() > 0 and patch.min() < 0 and (patch.max() - patch.min()) > 0.5:
                out_img_array[i, j] = 1

    return out_img_array
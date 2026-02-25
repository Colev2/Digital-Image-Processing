import numpy as np
from fir_conv import fir_conv

def sobel_edge(
    in_img_array: np.ndarray,
    thres: float
) -> np.ndarray:
    """
    Εφαρμόζει ανίχνευση ακμών με χρήση των Sobel μασκών.
    Υπολογίζει το μέτρο του διανύσματος κλίσης και επιστρέφει δυαδική εικόνα ακμών.
    """

    # Ορισμός των Sobel μασκών για οριζόντιες (Gx) και κατακόρυφες (Gy) παραγώγους
    Gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float)

    Gy = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=float)

    # Θέση εκκίνησης στην εικόνα (πάνω αριστερά)
    in_origin = np.array([0, 0], dtype=int)

    # Κέντρο της μάσκας (δηλ. σημείο ευθυγράμμισης για συνέλιξη)
    mask_origin = np.array([1, 1], dtype=int)

    # Εφαρμογή συνέλιξης με Gx και Gy (υπολογισμός κλίσεων)
    gx, _ = fir_conv(in_img_array, Gx, in_origin, mask_origin)
    gy, _ = fir_conv(in_img_array, Gy, in_origin, mask_origin)

    # Υπολογισμός του μέτρου του διανύσματος κλίσης για κάθε pixel
    gradient = np.sqrt(gx**2 + gy**2)

    # Κατωφλίωση: κρατάμε μόνο τα pixels με έντονη μεταβολή (πραγματικές ακμές)
    out_img_array = (gradient > thres).astype(int)

    return out_img_array
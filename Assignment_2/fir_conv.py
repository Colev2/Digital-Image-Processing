import numpy as np

def fir_conv(
    in_img_array: np.ndarray,
    h: np.ndarray,
    in_origin: np.ndarray,
    mask_origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Υλοποιεί 2D συνέλιξη (FIR φίλτρο) ανάμεσα σε εικόνα και μάσκα, 
    επιστρέφοντας το αποτέλεσμα και τη νέα θέση του origin.
    """

    # Αντιστροφή μάσκας ως προς x και y (κλασική συνέλιξη αντί για συσχέτιση)
    h_flipped = np.flipud(np.fliplr(h))

    # Λήψη διαστάσεων εικόνας και μάσκας
    in_rows, in_cols = in_img_array.shape
    h_rows, h_cols = h.shape

    # Υπολογισμός ποσότητας padding (με βάση τη θέση του origin της μάσκας)
    pad_top = mask_origin[0]
    pad_bottom = h_rows - mask_origin[0] - 1
    pad_left = mask_origin[1]
    pad_right = h_cols - mask_origin[1] - 1

    # Επέκταση εικόνας με μηδενικά ώστε να μην "κόβεται" στα άκρα
    padded_img = np.pad(in_img_array,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='constant', constant_values=0)

    # Δημιουργία πίνακα εξόδου (ίδιων διαστάσεων με την αρχική εικόνα)
    out_img_array = np.zeros_like(in_img_array, dtype=float)

    # Πράξη συνέλιξης σε κάθε pixel της εικόνας
    for i in range(in_rows):
        for j in range(in_cols):
            # Επιλέγουμε το κατάλληλο παράθυρο (region) από την εικόνα
            region = padded_img[i:i + h_rows, j:j + h_cols]

            # Υπολογισμός dot product (άθροισμα γινομένων) με τη μάσκα
            out_img_array[i, j] = np.sum(region * h_flipped)

    # Νέα θέση αρχής στην έξοδο (αθροίζονται τα origin εικόνας + μάσκας)
    out_origin = in_origin + mask_origin

    return out_img_array, out_origin

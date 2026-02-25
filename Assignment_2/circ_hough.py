import numpy as np

def circ_hough(in_img_array: np.ndarray, R_max: float, dim: np.ndarray, V_min: int):
    """
    Εντοπισμός κύκλων με Circular Hough Transform.
    Επιστρέφει λίστα από κέντρα και ακτίνες κυκλων που συγκέντρωσαν τουλάχιστον V_min ψήφους.
    """

    # Διαστάσεις εικόνας εισόδου
    H, W = in_img_array.shape

    # Πλήθος διακριτών τιμών για x, y και r στον συσσωρευτή
    x_bins, y_bins, r_bins = dim

    # Αρχικοποίηση 3D συσσωρευτή ψήφων (accumulator)
    acc = np.zeros((x_bins, y_bins, r_bins), dtype=np.uint32)

    # Διακριτές τιμές για x, y (κέντρα) και r (ακτίνα)
    x_range = np.linspace(0, W - 1, x_bins)
    y_range = np.linspace(0, H - 1, y_bins)
    r_range = np.linspace(20, R_max, r_bins)  # αποφυγή πολύ μικρών ακτίνων (π.χ. < 20)

    # Γωνίες 0–360° σε 10° βήματα (36 σημεία στον κύκλο)
    theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)

    # Ανίχνευση ακμιακών pixels (λευκά pixels = 1)
    ys, xs = np.where(in_img_array == 1)

    # Για κάθε ακμιακό pixel, ψηφίζουμε πιθανά κέντρα κύκλων για κάθε ακτίνα
    for x, y in zip(xs, ys):
        for r_idx, r in enumerate(r_range):
            # Υπολογισμός υποψήφιων κέντρων (α, β) για το συγκεκριμένο pixel και ακτίνα
            a_vals = x - r * np.cos(theta)
            b_vals = y - r * np.sin(theta)

            # Μετατροπή πραγματικών συντεταγμένων σε δείκτες στον συσσωρευτή
            x_idx = np.floor(a_vals * (x_bins - 1) / (W - 1)).astype(int)
            y_idx = np.floor(b_vals * (y_bins - 1) / (H - 1)).astype(int)

            # Φίλτρο για αποφυγή υπερχείλισης (εκτός ορίων)
            valid = (x_idx >= 0) & (x_idx < x_bins) & (y_idx >= 0) & (y_idx < y_bins)

            # Αυξάνουμε τον αριθμό ψήφων στον συσσωρευτή για τα έγκυρα σημεία
            for xi, yi in zip(x_idx[valid], y_idx[valid]):
                acc[xi, yi, r_idx] += 1

    # Συλλογή αποτελεσμάτων: βρίσκουμε τους κύκλους που συγκέντρωσαν αρκετές ψήφους
    centers = []
    radii = []

    for xi in range(x_bins):
        for yi in range(y_bins):
            for ri in range(r_bins):
                if acc[xi, yi, ri] >= V_min:    # κύκλος εντοπίστηκε, διότι πέρασε τις ψήφους του κατωφλίου V_min
                    # Αντιστοίχιση δεικτών σε πραγματικές συντεταγμένες και ακτίνα
                    x_center = x_range[xi]
                    y_center = y_range[yi]
                    radius = r_range[ri]

                    centers.append([x_center, y_center])
                    radii.append(radius)

    return np.array(centers), np.array(radii)
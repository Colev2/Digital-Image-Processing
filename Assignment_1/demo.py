import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from hist_utils import calculate_hist_of_img
from hist_modif import perform_hist_eq, perform_hist_matching

# --- Βοηθητική συνάρτηση για φόρτωση και προεπεξεργασία εικόνας ---
def load_and_prepare_image(filename):
    """
    Φορτώνει μια εικόνα από αρχείο, την μετατρέπει σε grayscale αν χρειάζεται
    και κανονικοποιεί τις τιμές της στο διάστημα [0,1].
    """
    img = Image.open(filename)
    if img.mode != 'L':
        img = img.convert('L')  # Μετατροπή σε grayscale αν είναι έγχρωμη
    img_array = np.array(img).astype(float)
    if img_array.max() > 1.0:
        img_array /= 255.0  # Κανονικοποίηση αν οι τιμές είναι 0–255
    return img_array

# --- Λήψη αρχείων από τον χρήστη ---
# Ο χρήστης δίνει τα μονοπάτια των εικόνων εισόδου και αναφοράς
input_filename = input("Δώσε το μονοπάτι της εικόνας εισόδου στον υπολογιστή σου (π.χ C:/Users/Nikos/Pictures/input_image_name.png):")
ref_filename = input("Δώσε το μονοπάτι της εικόνας αναφοράς στον υπολογιστή σου (π.χ C:/Users/Nikos/Pictures/reference_image_name.png): ")

# --- Φόρτωση εικόνων ---
# Προετοιμασία των εικόνων ώστε να είναι έτοιμες για επεξεργασία
input_array = load_and_prepare_image(input_filename)
ref_array = load_and_prepare_image(ref_filename)

# --- Modes ---
# Καθορισμός διαθέσιμων modes λειτουργίας (αλγορίθμων)
modes = ["greedy", "non-greedy", "post-disturbance"]

# --- Συνάρτηση plotting ---
def plot_comparison(original, processed, original_title, processed_title, fig_title):
    """
    Εμφανίζει την αρχική και την επεξεργασμένη εικόνα, μαζί με τα ιστογράμματά τους.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(fig_title, fontsize=16)

    # Εμφάνιση εικόνων
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title(original_title)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(processed, cmap='gray')
    axes[0, 1].set_title(processed_title)
    axes[0, 1].axis('off')

    # Υπολογισμός και εμφάνιση ιστογράμματος αρχικής εικόνας
    hist_orig = calculate_hist_of_img(original, return_normalized=True)
    axes[1, 0].bar(list(hist_orig.keys()), list(hist_orig.values()), width=0.005)
    axes[1, 0].set_title("Histogram of " + original_title)

    # Υπολογισμός και εμφάνιση ιστογράμματος επεξεργασμένης εικόνας
    hist_proc = calculate_hist_of_img(processed, return_normalized=True)
    axes[1, 1].bar(list(hist_proc.keys()), list(hist_proc.values()), width=0.005)
    axes[1, 1].set_title("Histogram of " + processed_title)

    plt.tight_layout()
    plt.show()

# --- Εξισορρόπηση Ιστογράμματος ---
# Εφαρμογή εξισορρόπησης ιστογράμματος για κάθε mode και εμφάνιση των αποτελεσμάτων
for mode in modes:
    eq_img = perform_hist_eq(input_array, mode)
    plot_comparison(input_array, eq_img,
                    "Original Image",
                    f"Equalized Image ({mode})",
                    f"Histogram Equalization - {mode.capitalize()} Mode")

# --- Προβολή Ιστογράμματος Αναφοράς ---
# Εμφάνιση μόνο του ιστογράμματος της εικόνας αναφοράς
ref_histogram = calculate_hist_of_img(ref_array, return_normalized=True)
plt.figure(figsize=(8,6))
plt.bar(list(ref_histogram.keys()), list(ref_histogram.values()), width=0.005)
plt.title("Histogram of Reference Image")
plt.xlabel("Intensity Value")
plt.ylabel("Normalized Frequency")
plt.grid(True)
plt.show()

# --- Αντιστοίχιση Ιστογράμματος ---
# Εφαρμογή αντιστοίχισης ιστογράμματος με βάση την εικόνα αναφοράς για κάθε mode
for mode in modes:
    matched_img = perform_hist_matching(input_array, ref_array, mode)
    plot_comparison(input_array, matched_img,
                    "Original Image",
                    f"Matched Image ({mode})",
                    f"Histogram Matching - {mode.capitalize()} Mode")

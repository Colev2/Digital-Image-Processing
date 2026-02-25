import numpy as np
import matplotlib.pyplot as plt
import cv2
from sobel_edge import sobel_edge
from log_edge import log_edge
from circ_hough import circ_hough


def main():
    # ========================== ΕΙΣΟΔΟΣ ΕΙΚΟΝΑΣ ==========================

    # Ζητάμε από τον χρήστη να δώσει το path της εικόνας
    img_path = input("Δώσε το path της εικόνας (π.χ.:  C:/Users/STAMATIS/Documents/Erg_2/basketball_large.png): ")

    print("Φόρτωση εικόνας...")

    # Φόρτωση εικόνας και μετατροπή της σε grayscale [0,1]
    img = cv2.imread(img_path)
    if img is None:
        print("Σφάλμα: Η εικόνα δεν φορτώθηκε. Έλεγξε το path και προσπάθησε ξανά.")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0

    # ========================== SOBEL ANALYSIS ==========================

    print(" Εκτελείται Sobel για διάφορα thresholds...")

    # Ορισμός τιμών threshold και αποθήκευση αποτελεσμάτων
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    points_counts = []
    sobel_results = []

    # Εκτέλεση sobel_edge για κάθε κατώφλι και μέτρηση ενεργών pixels
    for t in thresholds:
        print(f"   Threshold = {t}")
        sobel_img = sobel_edge(gray, t)
        sobel_results.append(sobel_img)
        points_counts.append(np.sum(sobel_img))

    # Εμφάνιση όλων των εικόνων ακμών για κάθε threshold
    print(" Προβολή Sobel για όλα τα thresholds...")
    plt.figure(figsize=(12, 6))
    for i, sobel_img in enumerate(sobel_results):
        plt.subplot(2, 3, i + 1)
        plt.imshow(sobel_img, cmap='gray')
        plt.title(f"thres={thresholds[i]}")
        plt.axis('off')
    plt.suptitle("Sobel Edge Detection (διάφορα thresholds)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Σχεδίαση γραφήματος: threshold vs πλήθος ακμιακών σημείων
    print(" Σχεδίαση γραφήματος Sobel...")
    plt.figure()
    plt.plot(thresholds, points_counts, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("Number of edge points")
    plt.title("Sobel Threshold Effect")
    plt.grid()
    plt.show()

    # ========================== LoG ANALYSIS ==========================

    # Εφαρμογή Gaussian blur πριν το LoG για μείωση θορύβου
    gray_blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)
    print(" Εκτελείται Laplacian of Gaussian (LoG)...")
    log_img = log_edge(gray_blur)

    # Εμφάνιση εικόνας ακμών από LoG
    print(" Προβολή LoG...")
    plt.figure()
    plt.imshow(log_img, cmap='gray')
    plt.title("LoG Edge Detection")
    plt.axis("off")
    plt.show()

    # ========================== RESIZE για Hough ==========================

    # Μείωση ανάλυσης για ταχύτερη εκτέλεση Hough
    img_small = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY) / 255.0



    
    # ========================== SOBEL + HOUGH ==========================

    # Ανίχνευση ακμών με Sobel σε μικρή εικόνα
    print("\n>>> Εκτελείται Sobel (thresh=0.5)...")
    edges = sobel_edge(gray_small, thres=0.5)

    # Ορισμός παραμέτρων Hough
    R_max = 300
    dim = np.array([100, 100, 50])  # διακριτοποίηση του συσσωρευτή
    V_min = [160, 180, 200, 220, 240]  # διάφορες τιμές ψήφων


    for V in V_min:
        print(f"\n>>> Εκτελείται Sobel και Hough με V_min = {V}")
        centers, radii = circ_hough(edges, R_max=R_max, dim=dim, V_min=V)

        result = img_small.copy()
        
        drawn = []  # Λίστα για αποθήκευση των ήδη σχεδιασμένων κέντρων, χρησιμοποιείται για αποφυγή επαναλαμβανόμενων ή πολύ κοντινών κύκλων
                    

        for (x, y), r in zip(centers, radii):
            print(f"x={x:.1f}, y={y:.1f}, r={r:.1f}")

            # Αν το κέντρο του νέου κύκλου είναι πολύ κοντά σε ήδη σχεδιασμένο (π.χ. < 15 pixels),
            # αγνοούμε τον κύκλο για να αποφύγουμε επικάλυψη ή διπλή σχεδίαση στην ίδια περιοχή
            if any(np.hypot(x - xc, y - yc) < 15 for xc, yc in drawn):
                continue

            # Καταχωρούμε το νέο κέντρο στη λίστα, ώστε να χρησιμοποιηθεί στον επόμενο έλεγχο
            drawn.append((x, y))

            # Σχεδιάζουμε τον κύκλο επάνω στην εικόνα
            cv2.circle(result, (int(x), int(y)), int(r), (0, 255, 0), 2)

        print(f">>> Εντοπίστηκαν {len(radii)} κύκλοι.")
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f"Sobel + Hough (V_min = {V})")
        plt.axis("off")
        plt.show()

    
    # ========================== LoG + HOUGH ==========================
    
    print("\n>>> Εκτελείται Laplacian of Gaussian (LoG)...")
    edges = log_edge(gray_small)

    V_min = [550, 560, 570, 575, 578]  # μεγαλύτερες τιμές λόγω θορύβου

    for V in V_min:
        print(f"\n>>> Εκτελείται LoG και Hough με V_min = {V}")
        centers, radii = circ_hough(edges, R_max=R_max, dim=dim, V_min=V)

        result = img_small.copy()
        drawn = []

        for (x, y), r in zip(centers, radii):
            print(f"x={x:.1f}, y={y:.1f}, r={r:.1f}")
            if any(np.hypot(x - xc, y - yc) < 15 for xc, yc in drawn):
                continue
            drawn.append((x, y))
            cv2.circle(result, (int(x), int(y)), int(r), (0, 255, 0), 2)

        print(f">>> Εντοπίστηκαν {len(radii)} κύκλοι.")
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f"LoG + Hough (V_min = {V})")
        plt.axis("off")
        plt.show()

    print("\nΤέλος εργασίας")

if __name__ == "__main__":
    main()
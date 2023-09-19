import cv2
import numpy as np
import matplotlib.pyplot as plt

# Girdi görüntüsünü yükle
input_image = cv2.imread('Rice.png', 0)  # Gri tonlamalı olarak yükle

# Pencere boyutları
window_sizes = [3, 5, 7, 9, 11]

# Adaptif eşikleme işlemini uygula ve sonuçları hesapla
adaptive_results = []
for size in window_sizes:
    # Adaptif eşikleme işlemi için pencere boyutu ve eşikleme türü belirle
    thresholded = cv2.adaptiveThreshold(input_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, size, np.mean(window_sizes))

    # Açılım işlemi
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, se)

    # Girdi görüntüsünden açılım sonucunu çıkar
    adaptive = cv2.subtract(thresholded, opening)
    adaptive_results.append(adaptive)

# Bağlantılı bileşen analizi yap
total_rice_count = []
total_rice_area = []
average_rice_area = []

for i, window in enumerate(adaptive_results):
    # Bağlantılı bileşenleri etiketle
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(window, connectivity=8)

    # İlk bileşeni dışarıda tut (arka plan)
    num_labels -= 1

    # Bileşen analizi için toplam pirinç sayısı ve alanı hesapla
    rice_count = num_labels
    rice_area = np.sum(stats[1:, cv2.CC_STAT_AREA])

    total_rice_count.append(rice_count)
    total_rice_area.append(rice_area)

    # Ortalama pirinç alanını hesapla
    average_area = rice_area / rice_count
    average_rice_area.append(average_area)

    print(f"Adaptif Eşikleme (Window Size={window_sizes[i]}): Rice Count={rice_count}, Total Rice Area={rice_area}, "
          f"Average Rice Area={average_area}")
# Grafik çizdirme
plt.figure(figsize=(10, 6))
plt.errorbar(window_sizes, total_rice_count, yerr=np.sqrt(total_rice_count), fmt='o-', label='Rice Count')
plt.errorbar(window_sizes, total_rice_area, yerr=np.sqrt(total_rice_area), fmt='o-', label='Total Rice Area')
plt.errorbar(window_sizes, average_rice_area, yerr=np.sqrt(average_rice_area), fmt='o-', label='Average Rice Area')
plt.xlabel('Window Size')
plt.ylabel('Measurements')
plt.title('Measurements vs Window Size')
plt.legend()
plt.grid(True)
plt.show()

# Sonuçları görüntüle
for i, window in enumerate(adaptive_results):
    cv2.imshow(f"Adaptif Eşikleme (Window Size={window_sizes[i]})", window)

cv2.waitKey(0)
cv2.destroyAllWindows()

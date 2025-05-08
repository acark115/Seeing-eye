import cv2
import numpy as np
import subprocess
import time

# Sesli konuşma fonksiyonu
def speak(text):
    subprocess.run(["espeak", "-v", "tr", "-s", "150", text])

# Türkçe etiketler
turkish_labels = {
    "person": "insan",
    "cat": "kedi",
    "dog": "köpek"
}

# İlgi duyulan nesneler
interested_objects = ["person", "cat", "dog"]

# Zaman takibi
last_spoken_time_color = {"green": 0, "red": 0}
last_spoken_object = {}
speak_interval = 5  # saniye

# Model dosyaları ve sınıf isimleri
classNames = []
classFile = "/home/fsociety/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/fsociety/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/fsociety/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

# Modeli yükle
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Kamera başlat
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Ana döngü
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # HSV'ye çevir
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Yeşil renk maskesi
    lower_green = np.array([35, 70, 50])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Kırmızı renk maskesi
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    green_count = cv2.countNonZero(green_mask)
    red_count = cv2.countNonZero(red_mask)

    # Renk uyarıları
    if green_count > red_count and green_count > 3000:
        cv2.putText(frame, "Yesil algilandi - YEŞİL IŞIK GEC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if current_time - last_spoken_time_color["green"] > speak_interval:
            speak("geç")
            last_spoken_time_color["green"] = current_time

    elif red_count > 3000:
        cv2.putText(frame, "Kirmizi algilandi - KIRMIZI IŞIK BEKLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if current_time - last_spoken_time_color["red"] > speak_interval:
            speak("bekle")
            last_spoken_time_color["red"] = current_time

    # Nesne algılama
    classIds, confs, bbox = net.detect(frame, confThreshold=0.45, nmsThreshold=0.2)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in interested_objects:
                x, y, w, h = box
                box_height = h
                turkish_name = turkish_labels.get(className, className)

                if box_height > 300:
                    distance_warning = "çok yakın, dikkatli ol"
                elif box_height > 200:
                    distance_warning = "yakın"
                else:
                    distance_warning = ""

                cv2.rectangle(frame, box, (0, 255, 0), 2)
                cv2.putText(frame, f"{turkish_name}", (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if className not in last_spoken_object or (current_time - last_spoken_object[className]) > speak_interval:
                    if distance_warning:
                        speak(f"{turkish_name} {distance_warning}")
                        print(f"Uyarı: {turkish_name} {distance_warning}")
                    else:
                        speak(turkish_name)
                        print(f"Algılandı: {turkish_name}")
                    last_spoken_object[className] = current_time

    # Görüntüyü göster
    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

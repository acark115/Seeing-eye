"""
import cv2
import numpy as np
import time
import os
from gtts import gTTS
import pygame
import threading

# Pygame ses sistemi başlat
pygame.mixer.init()

# Sesli uyarı çalma fonksiyonu (Threading ile arka planda çalıştırmak için)
def play_audio(text, key):
    filename = f"temp_{key}.mp3"
    if not os.path.exists(filename):  # Eğer ses dosyası yoksa oluştur
        try:
            tts = gTTS(text=text, lang='tr')
            tts.save(filename)
        except Exception as e:
            print(f"gTTS hatası: {e}")
            return
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Ses bitene kadar bekle
            time.sleep(0.1)
    except Exception as e:
        print(f"Ses çalma hatası: {e}")

# Etiket çevirileri
turkish_labels = {
    "person": "insan",
    "cat": "kedi",
    "dog": "köpek"
}

# Mesafe türünü belirleme
def get_distance_type(height):
    if height > 300:
        return "çok yakın"
    elif height > 200:
        return "yakın"
    return None  # çok uzakta

# Son konuşma zamanlarını tutma
last_spoken_time = {}
speak_interval = 5  # saniye

# Model dosyaları
classFile = "/home/fsociety/Desktop/Object_Detection_Files/coco.names"
configPath = "/home/fsociety/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/fsociety/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

# Etiketleri okuma
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Modeli yükleme
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)  # Çözünürlük düşürüldü
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Kamerayı başlatma
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Kamera çözünürlüğü 640x480
cap.set(4, 480)

print("Kamera başlatıldı. ESC ile çıkabilirsiniz.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    current_time = time.time()

    # Nesne tespiti
    classIds, confs, bbox = net.detect(frame, confThreshold=0.45, nmsThreshold=0.2)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            x, y, w, h = box

            distance_type = get_distance_type(h)
            if not distance_type:
                continue  # çok uzakta

            # Etiket ve konuşma mesajı
            if className in turkish_labels:
                turkish_name = turkish_labels[className]
                message = f"{distance_type} bir {turkish_name} var"
                key = f"{className}_{distance_type}"
            else:
                turkish_name = "cisim"
                message = f"{distance_type} bir cisim var"
                key = f"cisim_{distance_type}"

            # Ses dosyası çalma kontrolü ve threading
            if key not in last_spoken_time or (current_time - last_spoken_time[key]) > speak_interval:
                print(f"Sesli uyarı: {message}")
                # Ses çalmayı yeni bir thread ile başlat
                threading.Thread(target=play_audio, args=(message, key)).start()
                last_spoken_time[key] = current_time

            # Ekrana çizim
            cv2.rectangle(frame, box, (0, 255, 0), 2)
            cv2.putText(frame, turkish_name, (x + 10, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Kamera", frame)

    # ESC tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == 27:
        print("ESC tuşuna basıldı. Program sonlandırılıyor...")
        break

# Kamera ve pencereyi kapatma
cap.release()
cv2.destroyAllWindows()

# Geçici ses dosyalarını silme
print("Geçici ses dosyaları temizleniyor...")
for file in os.listdir():
    if file.startswith("temp_") and file.endswith(".mp3"):
        try:
            os.remove(file)
            print(f"Silindi: {file}")
        except Exception as e:
            print(f"Dosya silinemedi: {file}, Hata: {e}")

print("Program başarıyla kapatıldı.")



"""
"""
import cv2
import numpy as np
import time
import subprocess

# ------------------- SESLİ KONUŞMA FONKSİYONU -------------------
def speak(text):
    subprocess.Popen(['espeak', '-v', 'tr', text])

# ------------------- NESNE TANIMA -------------------
turkish_labels = {
    "person": "insan",
    "cat": "kedi",
    "dog": "köpek"
}

def get_distance_type(height):
    if height > 300:
        return "çok yakın"
    elif height > 200:
        return "yakın"
    return None

last_spoken_time = {}
speak_interval = 5  # saniye

# ------------------- MODEL YOLLARI -------------------
classFile = "/home/fsociety/Desktop/Object_Detection_Files/coco.names"
configPath = "/home/fsociety/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/fsociety/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

# Etiketleri oku
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Modeli yükle
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# ------------------- KAMERA -------------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Kamera başlatıldı. ESC ile çıkabilirsiniz.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    current_time = time.time()

    # ---------- NESNE TESPİTİ ----------
    classIds, confs, bbox = net.detect(frame, confThreshold=0.45, nmsThreshold=0.2)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            x, y, w, h = box

            distance_type = get_distance_type(h)
            if not distance_type:
                continue

            if className in turkish_labels:
                turkish_name = turkish_labels[className]
                message = f"{distance_type} bir {turkish_name} var"
                key = f"{className}_{distance_type}"
            else:
                turkish_name = "cisim"
                message = f"{distance_type} bir cisim var"
                key = f"cisim_{distance_type}"

            if key not in last_spoken_time or (current_time - last_spoken_time[key]) > speak_interval:
                print(f"Sesli uyarı: {message}")
                speak(message)
                last_spoken_time[key] = current_time

            # Ekrana kutu çiz
            cv2.rectangle(frame, box, (0, 255, 0), 2)
            cv2.putText(frame, turkish_name, (x + 10, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ---------- RENK ALGILAMA ----------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı renk aralığı
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Yeşil renk aralığı
    lower_green = np.array([40, 70, 70])
    upper_green = np.array([90, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    red_detected = cv2.countNonZero(mask_red) > 500
    green_detected = cv2.countNonZero(mask_green) > 500

    if red_detected:
        if 'red' not in last_spoken_time or (current_time - last_spoken_time['red']) > speak_interval:
            print("Kırmızı renk algılandı: Lütfen bekleyin")
            speak("Lütfen bekleyin")
            last_spoken_time['red'] = current_time
        cv2.putText(frame, "Lütfen bekleyin", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    elif green_detected:
        if 'green' not in last_spoken_time or (current_time - last_spoken_time['green']) > speak_interval:
            print("Yeşil renk algılandı: Geçebilirsiniz")
            speak("Geçebilirsiniz")
            last_spoken_time['green'] = current_time
        cv2.putText(frame, "Geçebilirsiniz", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Görüntüyü göster
    cv2.imshow("Kamera", frame)

    # ESC tuşu ile çık
    if cv2.waitKey(1) & 0xFF == 27:
        print("ESC tuşuna basıldı. Program sonlandırılıyor...")
        break

# ------------------- KAPAT -------------------
cap.release()
cv2.destroyAllWindows()
print("Program başarıyla kapatıldı.")




import cv2
import numpy as np
import time
import os
from gtts import gTTS
import pygame
import threading

pygame.mixer.init()

def play_audio(text, key):
    filename = f"temp_{key}.mp3"
    if not os.path.exists(filename):
        try:
            tts = gTTS(text=text, lang='tr')
            tts.save(filename)
        except Exception as e:
            print(f"gTTS hatası: {e}")
            return
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"Ses çalma hatası: {e}")

turkish_labels = {
    "person": "insan",
    "cat": "kedi",
    "dog": "köpek"
}

def get_distance_type(height):
    if height > 300:
        return "çok yakın"
    elif height > 200:
        return "yakın"
    return None

last_spoken_time = {}
speak_interval = 5  # saniye

classFile = "/home/fsociety/Desktop/Object_Detection_Files/coco.names"
configPath = "/home/fsociety/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/fsociety/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Kamera başlatıldı. ESC ile çıkabilirsiniz.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    current_time = time.time()

    # ------------------------------
    # 🔴🟢 Kırmızı ve Yeşil ışık tespiti (HSV renk aralığı ile)
    # ------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı için HSV aralıkları
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = red_mask1 + red_mask2

    # Yeşil için HSV aralığı
    green_lower = np.array([40, 70, 70])
    green_upper = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_detected = cv2.countNonZero(red_mask) > 1000
    green_detected = cv2.countNonZero(green_mask) > 1000

    # Renk mesajı konuşma (öncelik kırmızıda)
    if red_detected:
        key = "red_light"
        if key not in last_spoken_time or (current_time - last_spoken_time[key]) > speak_interval:
            threading.Thread(target=play_audio, args=("Kırmızı ışıkta dur", key)).start()
            last_spoken_time[key] = current_time
            cv2.putText(frame, "KIRMIZI IŞIK - DUR", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    elif green_detected:
        key = "green_light"
        if key not in last_spoken_time or (current_time - last_spoken_time[key]) > speak_interval:
            threading.Thread(target=play_audio, args=("Yeşil ışıkta geç", key)).start()
            last_spoken_time[key] = current_time
            cv2.putText(frame, "YESİL IŞIK - GEÇ", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # ------------------------------
    # 🔍 Nesne tespiti
    # ------------------------------
    classIds, confs, bbox = net.detect(frame, confThreshold=0.45, nmsThreshold=0.2)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            x, y, w, h = box
            distance_type = get_distance_type(h)
            if not distance_type:
                continue

            if className in turkish_labels:
                turkish_name = turkish_labels[className]
                message = f"{distance_type} bir {turkish_name} var"
                key = f"{className}_{distance_type}"
            else:
                turkish_name = "cisim"
                message = f"{distance_type} bir cisim var"
                key = f"cisim_{distance_type}"

            if key not in last_spoken_time or (current_time - last_spoken_time[key]) > speak_interval:
                print(f"Sesli uyarı: {message}")
                threading.Thread(target=play_audio, args=(message, key)).start()
                last_spoken_time[key] = current_time

            cv2.rectangle(frame, box, (0, 255, 0), 2)
            cv2.putText(frame, turkish_name, (x + 10, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("ESC tuşuna basıldı. Program sonlandırılıyor...")
        break

cap.release()
cv2.destroyAllWindows()

print("Geçici ses dosyaları temizleniyor...")
for file in os.listdir():
    if file.startswith("temp_") and file.endswith(".mp3"):
        try:
            os.remove(file)
            print(f"Silindi: {file}")
        except Exception as e:
            print(f"Dosya silinemedi: {file}, Hata: {e}")

print("Program başarıyla kapatıldı.")

"""



import cv2
import numpy as np
import time
import os
from gtts import gTTS
import pygame
import threading
from pydub import AudioSegment


pygame.mixer.init()

def play_audio(text, key):
    mp3_file = f"temp_{key}.mp3"
    wav_file = f"temp_{key}.wav"

    if not os.path.exists(wav_file):
        try:
            # MP3 dosyasını oluştur
            tts = gTTS(text=text, lang='tr')
            tts.save(mp3_file)

            # MP3 -> WAV dönüşümü
            sound = AudioSegment.from_file(mp3_file, format="mp3")
            sound.export(wav_file, format="wav")
        except Exception as e:
            print(f"Ses oluşturma hatası: {e}")
            return

    try:
        pygame.mixer.music.load(wav_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"Ses çalma hatası: {e}")


"""
pygame.mixer.init()

def play_audio(text, key):
    filename = f"temp_{key}.mp3"
    if not os.path.exists(filename):
        try:
            tts = gTTS(text=text, lang='tr')
            tts.save(filename)
        except Exception as e:
            print(f"gTTS hatası: {e}")
            return
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"Ses çalma hatası: {e}")
"""
turkish_labels = {
    "person": "insan",
    "cat": "kedi",
    "dog": "köpek"
}

def get_distance_type(height):
    if height > 300:
        return "çok yakın"
    elif height > 200:
        return "yakın"
    return None

last_spoken_time = {}
speak_interval = 5  # saniye

classFile = "/home/fsociety/Desktop/Object_Detection_Files/coco.names"
configPath = "/home/fsociety/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/fsociety/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Kamera başlatıldı. ESC ile çıkabilirsiniz.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    current_time = time.time()

    # ------------------------------
    # 🔴🟢 Kırmızı ve Yeşil ışık tespiti (HSV renk aralığı ile)
    # ------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı için HSV aralıkları
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = red_mask1 + red_mask2

    # Yeşil için HSV aralığı
    green_lower = np.array([40, 70, 70])
    green_upper = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_detected = cv2.countNonZero(red_mask) > 1000
    green_detected = cv2.countNonZero(green_mask) > 1000

    # Renk mesajı konuşma (öncelik kırmızıda)
    if red_detected:
        key = "red_light"
        if key not in last_spoken_time or (current_time - last_spoken_time[key]) > speak_interval:
            threading.Thread(target=play_audio, args=("Kırmızı ışıkta dur", key)).start()
            last_spoken_time[key] = current_time
            cv2.putText(frame, "KIRMIZI IŞIK - DUR", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    elif green_detected:
        key = "green_light"
        if key not in last_spoken_time or (current_time - last_spoken_time[key]) > speak_interval:
            threading.Thread(target=play_audio, args=("Yeşil ışıkta geç", key)).start()
            last_spoken_time[key] = current_time
            cv2.putText(frame, "YESİL IŞIK - GEÇ", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # ------------------------------
    # 🔍 Nesne tespiti
    # ------------------------------
    classIds, confs, bbox = net.detect(frame, confThreshold=0.45, nmsThreshold=0.2)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            x, y, w, h = box
            distance_type = get_distance_type(h)
            if not distance_type:
                continue

            if className in turkish_labels:
                turkish_name = turkish_labels[className]
                message = f"{distance_type} bir {turkish_name} var"
                key = f"{className}_{distance_type}"
            else:
                turkish_name = "cisim"
                message = f"{distance_type} bir cisim var"
                key = f"cisim_{distance_type}"

            if key not in last_spoken_time or (current_time - last_spoken_time[key]) > speak_interval:
                print(f"Sesli uyarı: {message}")
                threading.Thread(target=play_audio, args=(message, key)).start()
                last_spoken_time[key] = current_time

            cv2.rectangle(frame, box, (0, 255, 0), 2)
            cv2.putText(frame, turkish_name, (x + 10, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("ESC tuşuna basıldı. Program sonlandırılıyor...")
        break

cap.release()
cv2.destroyAllWindows()

print("Geçici ses dosyaları temizleniyor...")
for file in os.listdir():
    if file.startswith("temp_") and file.endswith(".mp3"):
        try:
            os.remove(file)
            print(f"Silindi: {file}")
        except Exception as e:
            print(f"Dosya silinemedi: {file}, Hata: {e}")

print("Program başarıyla kapatıldı.")

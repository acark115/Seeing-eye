"""import cv2
import subprocess  # Sesli okuma için
import time        # Aynı nesneyi tekrar tekrar söylememesi için

classNames = []
classFile = "/home/fsociety/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/fsociety/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/fsociety/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Konuşulan nesneleri ve zamanlarını tutmak için sözlük
last_spoken = {}

def speak(text):
    subprocess.run(["espeak", "-s", "150", text])

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    if len(objects) == 0: 
        objects = classNames
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img, 0.45, 0.2)

        current_time = time.time()

        for box, name in objectInfo:
            # Eğer daha önce söylenmemişse ya da üzerinden 5 saniye geçmişse tekrar söyle
            if name not in last_spoken or (current_time - last_spoken[name]) > 5:
                print(f"Detected: {name}")
                speak(name)
                last_spoken[name] = current_time

        cv2.imshow("Output", img)
        if cv2.waitKey(1) == 27:  # ESC tuşuna basılırsa çık
            break

    cap.release()
    cv2.destroyAllWindows()
"""
"""import cv2
import subprocess
import time

# Türkçe sınıf isimleri sözlüğü (İstediğin kadar genişletebilirsin)
turkish_labels = {
    "person": "insan",
    "bicycle": "bisiklet",
    "car": "araba",
    "motorcycle": "motosiklet",
    "airplane": "uçak",
    "bus": "otobüs",
    "train": "tren",
    "truck": "kamyon",
    "boat": "tekne",
    "traffic light": "trafik ışığı",
    "fire hydrant": "yangın musluğu",
    "stop sign": "dur işareti",
    "parking meter": "parkmetre",
    "bench": "bank",
    "bird": "kuş",
    "cat": "kedi",
    "dog": "köpek",
    "horse": "at",
    "sheep": "koyun",
    "cow": "inek",
    "elephant": "fil",
    "bear": "ayı",
    "zebra": "zebra",
    "giraffe": "zürafa",
    "backpack": "sırt çantası",
    "umbrella": "şemsiye",
    "handbag": "el çantası",
    "tie": "kravat",
    "suitcase": "bavul",
    "frisbee": "frizbi",
    "skis": "kayaklar",
    "snowboard": "snowboard",
    "sports ball": "top",
    "kite": "uçurtma",
    "baseball bat": "beyzbol sopası",
    "baseball glove": "beyzbol eldiveni",
    "skateboard": "kaykay",
    "surfboard": "sörf tahtası",
    "tennis racket": "tenis raketi",
    "bottle": "şişe",
    "wine glass": "şarap bardağı",
    "cup": "fincan",
    "fork": "çatal",
    "knife": "bıçak",
    "spoon": "kaşık",
    "bowl": "kase",
    "banana": "muz",
    "apple": "elma",
    "sandwich": "sandviç",
    "orange": "portakal",
    "broccoli": "brokoli",
    "carrot": "havuç",
    "hot dog": "sosisli",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "pasta",
    "chair": "sandalye",
    "couch": "kanepe",
    "potted plant": "saksı bitkisi",
    "bed": "yatak",
    "dining table": "yemek masası",
    "tv": "televizyon",
    "laptop": "laptop",
    "mouse": "fare",
    "remote": "kumanda",
    "keyboard": "klavye",
    "cell phone": "cep telefonu",
    "microwave": "mikrodalga",
    "oven": "fırın",
    "toaster": "ekmek kızartma makinesi",
    "sink": "lavabo",
    "refrigerator": "buzdolabı",
    "book": "kitap",
    "clock": "saat",
    "vase": "vazo",
    "scissors": "makas",
    "teddy bear": "oyuncak ayı",
    "hair drier": "saç kurutma makinesi",
    "toothbrush": "diş fırçası"
}

# Seslendirme fonksiyonu (Türkçe)
def speak(text):
    subprocess.run(["espeak", "-v", "tr", "-s", "150", text])

# Sınıf adlarını yükle
classNames = []
classFile = "/home/fsociety/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Model dosyaları
configPath = "/home/fsociety/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/fsociety/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

# Model yükleme
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Aynı nesneyi sürekli söylememek için zaman takibi
last_spoken = {}

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    if len(objects) == 0:
        objects = classNames
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

# Ana döngü
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img, 0.45, 0.2)

        current_time = time.time()
        for box, name in objectInfo:
            turkish_name = turkish_labels.get(name, name)  # Türkçeye çevir
            if name not in last_spoken or (current_time - last_spoken[name]) > 5:
                print(f"Algılandı: {turkish_name}")
                speak(turkish_name)
                last_spoken[name] = current_time

        cv2.imshow("Kamera", img)
        if cv2.waitKey(1) == 27:  # ESC tuşuyla çık
            break

    cap.release()
    cv2.destroyAllWindows()
"""
import cv2
import subprocess
import time

# Türkçe sınıf isimleri sözlüğü
turkish_labels = {
    "person": "insan",
    "bicycle": "bisiklet",
    "car": "araba",
    "motorcycle": "motosiklet",
    "airplane": "uçak",
    "bus": "otobüs",
    "train": "tren",
    "truck": "kamyon",
    "boat": "tekne",
    "traffic light": "trafik ışığı",
    "fire hydrant": "yangın musluğu",
    "stop sign": "dur işareti",
    "parking meter": "parkmetre",
    "bench": "bank",
    "bird": "kuş",
    "cat": "kedi",
    "dog": "köpek",
    "horse": "at",
    "sheep": "koyun",
    "cow": "inek",
    "elephant": "fil",
    "bear": "ayı",
    "zebra": "zebra",
    "giraffe": "zürafa",
    "backpack": "sırt çantası",
    "umbrella": "şemsiye",
    "handbag": "el çantası",
    "tie": "kravat",
    "suitcase": "bavul",
    "frisbee": "frizbi",
    "skis": "kayaklar",
    "snowboard": "snowboard",
    "sports ball": "top",
    "kite": "uçurtma",
    "baseball bat": "beyzbol sopası",
    "baseball glove": "beyzbol eldiveni",
    "skateboard": "kaykay",
    "surfboard": "sörf tahtası",
    "tennis racket": "tenis raketi",
    "bottle": "şişe",
    "wine glass": "şarap bardağı",
    "cup": "fincan",
    "fork": "çatal",
    "knife": "bıçak",
    "spoon": "kaşık",
    "bowl": "kase",
    "banana": "muz",
    "apple": "elma",
    "sandwich": "sandviç",
    "orange": "portakal",
    "broccoli": "brokoli",
    "carrot": "havuç",
    "hot dog": "sosisli",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "pasta",
    "chair": "sandalye",
    "couch": "kanepe",
    "potted plant": "saksı bitkisi",
    "bed": "yatak",
    "dining table": "yemek masası",
    "tv": "televizyon",
    "laptop": "laptop",
    "mouse": "fare",
    "remote": "kumanda",
    "keyboard": "klavye",
    "cell phone": "cep telefonu",
    "microwave": "mikrodalga",
    "oven": "fırın",
    "toaster": "ekmek kızartma makinesi",
    "sink": "lavabo",
    "refrigerator": "buzdolabı",
    "book": "kitap",
    "clock": "saat",
    "vase": "vazo",
    "scissors": "makas",
    "teddy bear": "oyuncak ayı",
    "hair drier": "saç kurutma makinesi",
    "toothbrush": "diş fırçası"
}

# Sesli konuşma fonksiyonu
def speak(text):
    subprocess.run(["espeak", "-v", "tr", "-s", "150", text])

# Sınıf adlarını yükle
classNames = []
classFile = "/home/fsociety/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Model dosyaları
configPath = "/home/fsociety/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/fsociety/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

# Modeli yükle
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Daha önce söylenen nesneleri hatırlamak için sözlük
last_spoken = {}

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    if len(objects) == 0:
        objects = classNames
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

# Ana döngü
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img, 0.45, 0.2)

        current_time = time.time()

        for box, name in objectInfo:
            turkish_name = turkish_labels.get(name, name)

            # Mesafe tahmini için kutu yüksekliği
            x, y, w, h = box
            box_height = h

            if box_height > 300:
                distance_warning = "çok yakın, dikkatli ol"
            elif box_height > 200:
                distance_warning = "yakın"
            else:
                distance_warning = ""

            # Eğer yeni algılandıysa veya belli bir süre geçtiyse sesli bildir
            if name not in last_spoken or (current_time - last_spoken[name]) > 5:
                if distance_warning:
                    print(f"Uyarı: {turkish_name} {distance_warning}")
                    speak(f"{turkish_name} {distance_warning}")
                else:
                    print(f"Algılandı: {turkish_name}")
                    speak(turkish_name)

                last_spoken[name] = current_time

        cv2.imshow("Kamera", img)
        if cv2.waitKey(1) == 27:  # ESC tuşu
            break

    cap.release()
    cv2.destroyAllWindows()

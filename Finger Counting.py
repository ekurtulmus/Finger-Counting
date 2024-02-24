print("FINGER COUNTING") #(parmak sayma)
import cv2
import mediapipe as mp

# Kamera tanımlama
cap = cv2.VideoCapture(0)

# Görüntü boyutlarını ayarlama
cap.set(3, 640) #görüntü boyutunu ayarlamamızın sebebi işlemciyi ve belleği yormamak
cap.set(4, 480)

#mediapipe kütüphanesinden el tanıma modülü
mpHand = mp.solutions.hands
hands= mpHand.Hands()
mpDraw=mp.solutions.drawing_utils #elin üzerindeki iskelet görünümünü oluşturmmızı sağlar (kırmızı ve beyaz olanlar)
# Ana döngü

tipIds=[4,8,12,16,20] #parmak uçlarının indexleri
while True:
    # Kameradan görüntü al ve görüntüyü (img) döndür
    success, img = cap.read()
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #renkleri çevirdik

    results=hands.process(imgRGB) #el tanıma modülünü kullanarak el tespit işlemini gerçekleştir
    print(results.multi_hand_landmarks)

    lmList=[] #Eldeki landmark noktalarını saklamak için içi boş bir liste oluşturduk
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS) #el üzerindeki landmark noktalarını ve bağlantıları çizdik

            for id, lm in enumerate(handLms.landmark): #enumerate=bizim handLms de bulunan landmarkların xyz sini "lm" içine, id lerini de "id" içine yazıyor
                h,w,c=img.shape #yükseklik genişlik ve renk
                cx, cy=int(lm.x*w), int(lm.y*h) #Landmark noktasının piksel koordinatlarını hesaplar.
                lmList.append([id, cx, cy]) #yukardaki listeyi id,cx,cy (koordinat) ile dolduracak


    if len (lmList) !=0:
        fingers=[]

        #bas parmak
        if lmList[tipIds[0]][1] < lmList[tipIds[0] -1][1]: #bu sol el için geçerli. Sağ el saymak için "küçüktür" işaretini "büyüktür" ile değiştirmeliyiz
            fingers.append(1)
        else:
            fingers.append(0)

        #4 parmak
        for id in range(1, 5): #4 parmak ile baş parmağı ayırdık
            if lmList[tipIds[id]][2]<lmList[tipIds[id] -2 ][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalF=fingers.count(1)
        cv2.putText(img, str(totalF), (30,125), cv2.FONT_HERSHEY_PLAIN,10, (255,0,0),8)
        
    print(lmList)



    # Görüntüyü işle
    # Burada el tespiti ve işaretlenmesi işlemlerini gerçekleştirebilirsiniz.

    # Görüntüyü göster
    cv2.imshow("img", img)

    # 'q' tuşuna basarak döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Döngüyü sonlandır ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import time

# باز کردن وبکم
webcam = cv2.VideoCapture(1)

# تشخیص دست
mphands = mp.solutions.hands
hands = mphands.Hands()  # دست
mpDraw = mp.solutions.drawing_utils  # کشیدن خطوط


ptime = 0
while True:
    cap, img = webcam.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res = hands.process(imgRGB)

    # print(res)
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10, 100), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (0, 0, 255), 3)

    if res.multi_hand_landmarks:
        for handlmk in res.multi_hand_landmarks:
            h, w, c = img.shape  # جابجایی این قسمت از حلقه

            for id, lm in enumerate(handlmk.landmark):
                x, y = int(lm.x * w), int(lm.y * h)

                if id in [4, 8, 12, 16, 20]:
                    cv2.circle(img, (x, y), 5, (255, 0, 0), 5)

            mpDraw.draw_landmarks(img, handlmk, mphands.HAND_CONNECTIONS)

    cv2.imshow('HAND TRACK', img)
    cv2.waitKey(1)

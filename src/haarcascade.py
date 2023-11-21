import cv2

# Haar-like特徴分類器の読み込み
face_cascade = cv2.CascadeClassifier('others/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('others/haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)

# メインループ 
while True:
    ret, frame = cap.read()

    # グレースケール変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔を検知
    # faces = face_cascade.detectMultiScale(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
    for (x,y,w,h) in faces:
        # 検知した顔を矩形で囲む
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # 処理高速化のために顔の上半分を検出対象範囲とする
        eyes_gray = gray[y : y + int(h/2), x : x + w]
        eyes = eye_cascade.detectMultiScale(eyes_gray, scaleFactor=1.11, minNeighbors=3, minSize=(8, 8))

        for ex, ey, ew, eh in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 255, 0), 1)

        # 目の数が 0 の時(閉じている時)
        if len(eyes) == 0:
            cv2.putText(frame,"Sleepy eyes. Wake up!",(10,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2, cv2.LINE_AA)

    # 画像表示
    cv2.imshow('frame',frame)

    # 何らかのキーが押されたら終了 
    key = cv2.waitKey(1)
    if key != -1:
        break
 
# 終了処理 
cap.release()
cv2.destroyAllWindows()

# 参考
# https://qiita.com/mogamin/items/a65e2eaa4b27aa0a1c23
# モデルダウンロード先
# https://github.com/opencv/opencv/tree/master/data/haarcascades
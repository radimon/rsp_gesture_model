import cv2
import numpy as np
import tensorflow as tf

# 1. 載入訓練好的強大模型 (請依據實際檔名調整)
model_path = "gesture_model_mobilenet.h5"
model = tf.keras.models.load_model(model_path)

# 2. 定義類別名稱 (需與訓練時的資料夾順序一致，通常是按字母排序)
# 預設: ['Error', 'Paper', 'Rock', 'Scissors']
class_names = ['Error', 'Paper', 'Rock', 'Scissors'] 

# 3. 開啟樹莓派相機
cap = cv2.VideoCapture(0)

print("開始執行手勢辨識。請展示 10 次手勢並包含：Rock, Scissors, Paper, Error。")
print("按下 'q' 鍵可結束程式。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 畫面上顯示預測區域或提示
    h, w, _ = frame.shape
    
    # 影像前處理：裁切或縮放至模型要求的 128x128
    img = cv2.resize(frame, (128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # 擴展維度符合 Batch 格式 (1, 128, 128, 3)

    # 進行預測
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    class_idx = np.argmax(predictions[0])
    result_label = class_names[class_idx]
    confidence = 100 * np.max(predictions[0]) # 或是使用 score

    # 將預測結果即時畫在 OpenCV 視窗畫面上
    display_text = f"Gesture: {result_label} ({confidence:.2f}%)"
    cv2.putText(frame, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 顯示畫面
    cv2.imshow('Raspberry Pi Gesture Demo', frame)
    
    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

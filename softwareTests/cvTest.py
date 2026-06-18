import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 24)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    cv2.imshow('Video Feed', frame)
    
    # Must be 1 or higher; 0 freezes the loop waiting for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
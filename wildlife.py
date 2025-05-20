import cv2

# Load HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load video file
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening video file")
    exit()


width, height = 640, 480

# First frame for motion detection
ret, frame1 = cap.read()
if not ret:
    print("Error reading first frame")
    exit()

frame1 = cv2.resize(frame1, (width, height))
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    frame2 = cv2.resize(frame2, (width, height))
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Motion detection
    diff = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = any(cv2.contourArea(cnt) > 1000 for cnt in contours)

    if motion_detected:
        human_detected = False

        # Human detection using HOG
        boxes, weights = hog.detectMultiScale(frame2, winStride=(8, 8))
        if len(boxes) > 0:
            human_detected = True
            for (x, y, w, h) in boxes:
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame2, "Human", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # If motion but no human detected
        if not human_detected:
            cv2.putText(frame2, "Unknown Animal Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Wildlife Monitoring", frame2)
    gray1 = gray2.copy()

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

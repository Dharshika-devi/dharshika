import cv2
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
img = cv2.imread('images/car car image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plates = plate_cascade.detectMultiScale (gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
for (x, y, w, h) in plates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0),5)
cv2.imwrite("images/car car image_output.jpg",img)
cv2.imshow("License Plate Detection", img)
cv2.waitKey(0)
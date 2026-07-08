import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image = cv2.imread(os.path.join(script_dir, 'data/coin.jpg'))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 0)
edged = cv2.Canny(gray, 50, 150)

contours, hierarchy = cv2.findContours(
    edged, 
    cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)

out = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
cv2.drawContours(out, contours, -1, (0, 255, 128), 2)

image = cv2.hconcat([image, out])
cv2.imshow('canny image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


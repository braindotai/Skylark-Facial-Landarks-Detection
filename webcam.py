import cv2
from pprint import pprint
from core.detector import detect
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    result, output_img = detect(cv2_image = img, verbose = False)

    # pprint(result)
    cv2.imshow('img', output_img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
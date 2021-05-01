import cv2
from pprint import pprint
from core.detector import detect

image = cv2.imread('./test_images/image5.jpg')
# json_outputs, image_outputs = detect(cv2_image = image, save_to_file = True, output_json_dir = 'json_outputs', output_image_dir = 'image_outputs', verbose = True)
json_outputs, image_outputs = detect(cv2_image = image, save_to_file = False, verbose = True)

pprint(json_outputs)
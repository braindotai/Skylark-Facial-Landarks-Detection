# Face Landmarks Detection

## Usage

- Pytorch installation:
  1. If gpu is available then install pytorch based on your gpu support from [here](https://pytorch.org/get-started/locally/).
  2. If gpu is not available then install pytorch simply by running:

       `$ pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`

- `$ pip install -r requirements.txt`
- Then you can import and run `detect` function from `core` using.

```python
from core.detector import detect
```

This function is meant to be run on inference.


Arguments:
```

  cv2_image              : (numpy.ndarray) A cv2 image, obtained from `cv2.imread`.
                           (default: `None`)
    
  video_path             : (str)  Path to the video on which inference will be run.
                           (default: `None`)
  
  predictions_per_second : (int) Number of predictions the model will make in each second. Ex: if set to 6, and fps is 30, then images
                                 in every second will be divided in 6 blocks (each block containing 30/6 images), then the prediction will be
                                 made on middle image of each block, and that prediction will get copied to all other images within that block.
                           (default: 1)
  
  save_to_file           : (bool) Will save the outputs if set to `True`. Either the `output_json_dir` or `output_video_dir` is required when set
  								  to `True`.
                           (default: `False`)
  
  output_json_dir        : (str)  Directory in which json outputs will be saved.
                           (default: None)

  output_image_dir       : (str)  Directory in which image outputs will be saved. Pass this only when `cv2_image` is not `None`.
                           (default: None)
  
  output_video_dir       : (str)  Directory in which video outputs will be saved. Pass this only when `video_path` is not `None`.
                           (default: None)
  
  verbose                : (bool) Will display outputs if `True`.
                           (default: `False`)
```

## Sample Usage

Code:

```python

import cv2
from pprint import pprint
from core.detector import detect

image = cv2.imread('test_images/image1.png')
outputs = detect(cv2_image = image, save_to_file = True, output_json_dir = 'json_outputs', output_image_dir = 'image_outputs', verbose = True)

pprint(outputs)
```

Outputs:


```

Running initialization...
Device    : cpu
Precision : float32
Batch size: 4

Reading image...
Making predictions...

{'json_output': {1: {'chin': [(663, 322),
                              (671, 326),
                              (679, 327),
                              (687, 325),
                              (692, 319)],
                     'left_eye': [(688, 274),
                                  (691, 272),
                                  (694, 272),
                                  (697, 274),
                                  (694, 275),
                                  (691, 275)],
                     'left_eyebrow': [(685, 268),
                                      (690, 266),
                                      (695, 265),
                                      (700, 265),
                                      (703, 268)],
                     'left_jawline': [(698, 313),
                                      (702, 306),
                                      (704, 298),
                                      (705, 289),
                                      (705, 280),
                                      (705, 272)],
                     'lips': [(666, 303),
                              (673, 302),
                              (678, 302),
                              (681, 303),
                              (684, 302),
                              (687, 302),
                              (690, 303),
                              (687, 307),
                              (684, 309),
                              (680, 309),
                              (677, 309),
                              (672, 307),
                              (668, 304),
                              (677, 304),
                              (681, 305),
                              (684, 304),
                              (689, 304),
                              (683, 305),
                              (680, 305),
                              (677, 305)],
                     'nose': [(681, 273),
                              (682, 279),
                              (682, 284),
                              (683, 290),
                              (675, 294),
                              (678, 295),
                              (682, 296),
                              (685, 295),
                              (687, 294)],
                     'right_eye': [(660, 273),
                                   (664, 271),
                                   (667, 271),
                                   (670, 274),
                                   (667, 275),
                                   (663, 274)],
                     'right_eyebrow': [(653, 267),
                                       (658, 264),
                                       (664, 264),
                                       (670, 265),
                                       (676, 267)],
                     'right_jawline': [(641, 271),
                                       (642, 281),
                                       (643, 291),
                                       (645, 301),
                                       (649, 309),
                                       (655, 316)]},
                 2: {'chin': [(340, 329),
                              (347, 333),
                              (355, 334),
                              (362, 332),
                              (368, 327)],
                     'left_eye': [(362, 285),
                                  (365, 282),
                                  (369, 281),
                                  (372, 283),
                                  (369, 285),
                                  (365, 285)],
                     'left_eyebrow': [(359, 280),
                                      (364, 277),
                                      (369, 275),
                                      (375, 275),
                                      (379, 277)],
                     'left_jawline': [(373, 320),
                                      (378, 314),
                                      (381, 306),
                                      (382, 297),
                                      (383, 288),
                                      (383, 280)],
                     'lips': [(343, 314),
                              (348, 314),
                              (352, 314),
                              (355, 314),
                              (358, 313),
                              (361, 313),
                              (365, 313),
                              (361, 316),
                              (358, 318),
                              (355, 319),
                              (352, 319),
                              (348, 318),
                              (345, 315),
                              (352, 315),
                              (355, 315),
                              (358, 315),
                              (363, 313),
                              (358, 315),
                              (355, 316),
                              (352, 316)],
                     'nose': [(354, 285),
                              (355, 292),
                              (355, 298),
                              (356, 304),
                              (349, 306),
                              (352, 308),
                              (356, 308),
                              (359, 307),
                              (361, 305)],
                     'right_eye': [(334, 285),
                                   (338, 283),
                                   (342, 283),
                                   (345, 286),
                                   (341, 287),
                                   (337, 287)],
                     'right_eyebrow': [(327, 280),
                                       (332, 278),
                                       (337, 277),
                                       (343, 278),
                                       (348, 280)],
                     'right_jawline': [(317, 282),
                                       (318, 291),
                                       (320, 300),
                                       (322, 309),
                                       (326, 317),
                                       (332, 323)]},
                 3: {'chin': [(1519, 308),
                              (1528, 311),
                              (1539, 312),
                              (1548, 310),
                              (1556, 306)],
                     'left_eye': [(1544, 260),
                                  (1547, 258),
                                  (1551, 259),
                                  (1555, 260),
                                  (1551, 261),
                                  (1547, 261)],
                     'left_eyebrow': [(1538, 253),
                                      (1545, 251),
                                      (1552, 249),
                                      (1558, 251),
                                      (1563, 255)],
                     'left_jawline': [(1565, 301),
                                      (1571, 295),
                                      (1573, 286),
                                      (1573, 277),
                                      (1573, 268),
                                      (1573, 259)],
                     'lips': [(1521, 286),
                              (1528, 285),
                              (1532, 284),
                              (1536, 284),
                              (1539, 283),
                              (1544, 284),
                              (1550, 285),
                              (1544, 287),
                              (1540, 288),
                              (1536, 288),
                              (1533, 288),
                              (1528, 288),
                              (1523, 286),
                              (1532, 286),
                              (1536, 286),
                              (1539, 285),
                              (1548, 285),
                              (1539, 285),
                              (1536, 285),
                              (1533, 285)],
                     'nose': [(1533, 258),
                              (1534, 262),
                              (1535, 267),
                              (1535, 271),
                              (1528, 277),
                              (1532, 278),
                              (1536, 278),
                              (1539, 277),
                              (1542, 276)],
                     'right_eye': [(1512, 261),
                                   (1516, 259),
                                   (1520, 259),
                                   (1523, 261),
                                   (1520, 262),
                                   (1516, 262)],
                     'right_eyebrow': [(1503, 257),
                                       (1508, 253),
                                       (1514, 251),
                                       (1520, 251),
                                       (1527, 253)],
                     'right_jawline': [(1494, 263),
                                       (1495, 273),
                                       (1496, 282),
                                       (1498, 291),
                                       (1502, 299),
                                       (1509, 305)]},
                 4: {'chin': [(1019, 229),
                              (1029, 236),
                              (1040, 239),
                              (1050, 235),
                              (1059, 227)],
                     'left_eye': [(1040, 173),
                                  (1043, 171),
                                  (1047, 171),
                                  (1052, 171),
                                  (1048, 173),
                                  (1044, 173)],
                     'left_eyebrow': [(1033, 168),
                                      (1040, 165),
                                      (1047, 162),
                                      (1054, 163),
                                      (1060, 164)],
                     'left_jawline': [(1068, 218),
                                      (1074, 207),
                                      (1077, 194),
                                      (1076, 182),
                                      (1075, 169),
                                      (1074, 157)],
                     'lips': [(1021, 207),
                              (1026, 207),
                              (1031, 207),
                              (1035, 208),
                              (1039, 206),
                              (1044, 206),
                              (1050, 205),
                              (1045, 212),
                              (1040, 215),
                              (1036, 216),
                              (1032, 215),
                              (1027, 213),
                              (1023, 207),
                              (1032, 208),
                              (1035, 209),
                              (1039, 208),
                              (1048, 206),
                              (1039, 212),
                              (1036, 213),
                              (1032, 212)],
                     'nose': [(1030, 174),
                              (1031, 181),
                              (1032, 188),
                              (1033, 194),
                              (1025, 198),
                              (1030, 200),
                              (1034, 201),
                              (1038, 199),
                              (1041, 197)],
                     'right_eye': [(1010, 173),
                                   (1014, 173),
                                   (1017, 172),
                                   (1021, 173),
                                   (1017, 174),
                                   (1014, 174)],
                     'right_eyebrow': [(1000, 168),
                                       (1005, 165),
                                       (1011, 165),
                                       (1018, 166),
                                       (1024, 168)],
                     'right_jawline': [(991, 166),
                                       (992, 178),
                                       (993, 190),
                                       (996, 201),
                                       (1001, 212),
                                       (1009, 221)]}},

 'output': 'b1e2327e-a156-4bd9-9d73-9281b53f5dc7_output.jpg'}
```



# Author

## __Rishik Mourya__

Contact for any query contact at __rishik@skylarklabs.ai__
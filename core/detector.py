from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torchvision.transforms.functional as TF

import cv2
# from .face_detector import detect as face_detector
from moviepy.editor import VideoFileClip

import os
import json
import uuid
from functools import reduce

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
device = None
precision = None
batch_size = None
initialized = None
face_detector = None

def initialize():
    global model, device, batch_size, precision, face_detector, initialized

    face_detector = cv2.CascadeClassifier('face_detection.xml')

    print('Running initialization...')
    device = os.getenv('DEVICE', 'cuda')
    precision = os.getenv('PRECISION', 'float16')
    batch_size = os.getenv('BATCH_SIZE', 4)
    batch_size = int(batch_size) if isinstance(batch_size, str) else batch_size
    
    print(f'Device    : {device}')
    print(f'Precision : {precision}')
    print(f'Batch size: {batch_size}')

    if device == 'cuda':
        assert torch.cuda.is_available(), '\n\n"device" is set to "cuda", but no gpu is found :(\n'
    elif device != 'cpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model().to(device)

    if precision == 'float16':
        assert device == 'cuda', f'\n\nHalf precision is only available when device is set to "cuda", but got "{device}".\n'
        model = model.half()

    initialized = True


@torch.no_grad()
def detect(cv2_image = None, video_path = None, predictions_per_second = 2, save_to_file = False, output_json_dir = None, output_image_dir = None, output_video_dir = None, verbose = False):
    """
    This function is meant to be run on inference.
    Arguments:
    cv2_image        : (numpy.ndarray)  A cv2 image, obtained from `cv2.imread`.
                                        (default: `None`)
    
    video_path       : (str)  Path to the video on which inference will be run.
                              (default: `None`)

    predictions_per_second : (int) Number of predictions the model will make in each second. Ex: if set to 6, and fps is 30, then images
                                   in every second will be divided in 6 blocks (each block containing 30/6 images), then the prediction will be
                                   made on middle image of each block, and that prediction will get copied to all other images within that block.
                                   (default: 2)
        
    save_to_file     : (bool) Will save the outputs if set to `True`. Either the `output_json_dir` or `output_video_dir` is required when set to True.
                              (default: `False`)
    
    output_json_dir  : (str)  Directory in which json outputs will be saved.
                              (default: None)

    output_image_dir : (str)  Directory in which image outputs will be saved. Pass this only when `cv2_image` is not `None`.
                              (default: None)
    
    output_video_dir : (str)  Directory in which video outputs will be saved. Pass this only when `video_path` is not `None`.
                              (default: None)
    
    verbose          : (bool) Will display outputs if `True`.
                              (default: `False`)
    """

    assert cv2_image is not None or video_path is not None, f'\n\nAt least one of `cv2_image` or `video_path` is required, found both `None`.\n'
    assert not all([cv2_image is not None, video_path is not None]), f'\n\nCannot accept both `cv2_image` and `video_path`.\n'

    if save_to_file:
        assert output_json_dir or output_video_dir or output_image_dir, f'\n\nPath for at least one of `output_json_dir`, `output_video_dir` and `output_image_dir` is required when `save_to_file` is set to `True`, found both `None`.\n'
    else:
        assert not output_json_dir and not output_video_dir and not output_image_dir, f'\n\nValues for `output_json_dir`, `output_video_dir` and `output_image_dir` are not accepted when `save_to_file` is set to `False`.\n'
    
    global model, device, batch_size, precision, initialized

    if not initialized:
        initialize()
    
    result = {'output': None}
    
    json_warn = False

    if video_path:
        if verbose:
            print('\nReading video...')
        video = VideoFileClip(video_path)
        inference_file = video_path
    else:
        if verbose:
            print('\nReading image...')
        inference_file = str(uuid.uuid4()) + ".jpg"

    if save_to_file:
        inference_file_name = inference_file.split("/")[-1].split('.')[0]

        if output_json_dir:
            if video_path:
                json_path = os.path.join(output_json_dir, f'{inference_file_name}_pps_{predictions_per_second}_output.json')
            else:
                json_path = os.path.join(output_json_dir, f'{inference_file_name}_output.json')
            
            if not os.path.isdir(output_json_dir):
                os.mkdir(output_json_dir)

            elif os.path.split(json_path)[1] in os.listdir(output_json_dir):
                if verbose:
                    json_warn = True
        
        if output_image_dir:
            file_name = f'{inference_file_name}_output.jpg'
            image_output_path = os.path.join(output_image_dir, file_name)
            
            if not os.path.isdir(output_image_dir):
                os.mkdir(output_image_dir)

            elif os.path.split(image_output_path)[1] in os.listdir(output_image_dir):
                if verbose:
                    image_output_warn = True
            
            result['output'] = file_name

        if output_video_dir and video_path:
            file_name = f'{inference_file_name}_pps_{predictions_per_second}_output.mp4'
            video_output_path = os.path.join(output_video_dir, file_name)
            
            if not os.path.isdir(output_video_dir):
                os.mkdir(output_video_dir)
            elif os.path.split(video_output_path)[1] in os.listdir(output_video_dir):
                if verbose:
                    print(f'WARNING...: Replacing a video output file with name "{os.path.split(video_output_path)[1]}" which already exists in the "{output_video_dir}".')
                    # print('WARNING: Replacing a video output file with name', os.path.split(video_output_path)[1], 'which already exists in the', output_video_dir)
            
            result['output'] = file_name
        else:
            video_output_path = None

    if verbose:
        print('Making predictions...')

    if video_path:       
        json_output = inference(video, predictions_per_second, video_output_path, precision, verbose)
    else:
        output_image, _, json_output = inference_on_mid_index(cv2_image, precision, True)

        if output_image_dir:
            cv2.imwrite(image_output_path, output_image)

    result['json_output'] = json_output

    if json_warn:
        print(f'WARNING: Replacing a json output file with name "{os.path.split(json_path)[1]}" which already exists in the "{output_json_dir}".')

    if output_json_dir:
        with open(json_path, 'w') as file:
            json.dump(json_output, file)
    
    return result, output_image


def get_model():
    ckpt_dir = os.path.join(BASE_DIR, 'parameters')
    assert os.path.isdir(ckpt_dir) and 'XceptionNet.pt' in os.listdir(ckpt_dir), '\n\nNo pretrained model is found.\n'
    
    model = XceptionNet()
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'XceptionNet.pt'), map_location = torch.device('cpu')))
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()

    return model



def draw_landmarks_on_faces(image, faces_landmarks, reverse = False):
    color = [255, 153, 0]
    landmarks_dict = {}
    
    for i, (landmarks, (left, top, height, width)) in enumerate(faces_landmarks, 1):
        landmarks = landmarks.view(-1, 2)
        landmarks = (landmarks + 0.5)
        # landmarks = landmarks.numpy()

        landmarks_dict[i] = {'left_eye': [],
                             'right_eye': [],
                             'left_eyebrow': [],
                             'right_eyebrow': [],
                             'nose': [],
                             'lips': [],
                             'chin': [],
                             'left_jawline': [],
                             'right_jawline': []}
        
        for j, (x, y) in enumerate(landmarks, 1):
            # try:
            x = int((x.item() * width) + left)
            y = int((y.item() * height) + top)

            if 28 <= j <= 36:
                landmarks_dict[i]['nose'].append((x, y))
            elif 37 <= j <= 42:
                landmarks_dict[i]['right_eye'].append((x, y))
            elif 43 <= j <= 48:
                landmarks_dict[i]['left_eye'].append((x, y))
            elif 18 <= j <= 22:
                landmarks_dict[i]['right_eyebrow'].append((x, y))
            elif 23 <= j <= 27:
                landmarks_dict[i]['left_eyebrow'].append((x, y))
            elif 7 <= j <= 11:
                landmarks_dict[i]['chin'].append((x, y))
            elif 1 <= j <= 6:
                landmarks_dict[i]['right_jawline'].append((x, y))
            elif 12 <= j <= 17:
                landmarks_dict[i]['left_jawline'].append((x, y))
            elif 49 <= j <= 68:
                landmarks_dict[i]['lips'].append((x, y))

            cv2.circle(image, (x, y), 2, color, -1)
            # except:
            #     pass
    
    return image, landmarks_dict


def preprocess_image(image):
    # image = TF.to_pil_image(image)
    # image = TF.resize(image, (128, 128))
    try:
        image = cv2.resize(image, (128, 128))
        image = TF.to_tensor(image)
        return image
    except:
        return None



@torch.no_grad()
def inference_on_mid_index(frame, precision, reverse = False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_outputs = face_detector.detectMultiScale(gray, 1.1, 4)

    outputs = []
    batch_images = []
    batch_coordinates = []

    for i, (x, y, w, h) in enumerate(faces_outputs):
        # x, y = face_coordinates['x1'], face_coordinates['y1']
        # w = face_coordinates['x2'] - face_coordinates['x1']
        # h = face_coordinates['y2'] - face_coordinates['y1']

        coordinates = (x, y, h, w)

        crop_img = gray[y: y + h, x: x + w]
        preprocessed_image = preprocess_image(crop_img)

        if preprocessed_image != None and len(batch_images) < batch_size:
            batch_images.append(preprocessed_image)
            batch_coordinates.append(coordinates)

        elif len(batch_images):
            x = torch.stack(batch_images, axis = 0).to(device)
            x = (x - x.min())/(x.max() - x.min())
            x = (2 * x) - 1
            
            if precision == 'float16':
                x = x.half()

            batch_predictions = model(x)

            outputs.extend([(landmarks_predictions, coordinates) for landmarks_predictions, coordinates in zip(batch_predictions, batch_coordinates)])
            
            del batch_images
            del batch_coordinates

            if preprocessed_image != None:
                batch_images = [preprocessed_image]
                batch_coordinates = [coordinates]
            else:
                batch_images = []
                batch_coordinates = []

    if len(batch_images):
        x = torch.stack(batch_images, axis = 0).to(device)
        x = (x - x.min())/(x.max() - x.min())
        x = (2 * x) - 1
            
        if precision == 'float16':
            x = x.half()
        
        batch_predictions = model(x)

        outputs.extend([(landmarks_predictions, coordinates) for landmarks_predictions, coordinates in zip(batch_predictions, batch_coordinates)])
        
        del batch_images
        del batch_coordinates

    image, landmarks_dict = draw_landmarks_on_faces(frame, outputs, reverse)

    return image, outputs, landmarks_dict


def secondsToStr(seconds):
    return "%02d:%02d:%02d.%03d" % reduce(lambda ll,b : divmod(ll[0],b) + ll[1:], [(round(seconds*1000),),1000,60,60])


def get_blocks(video, predictions_per_second):
    blocks = [[]]
    counter = (video.fps // predictions_per_second)

    for frame_index, image in enumerate(video.iter_frames(), 1):
        if frame_index < counter:
            blocks[-1].append((frame_index, image))
        elif frame_index == counter:
            counter += (video.fps // predictions_per_second)
            blocks[-1].append((frame_index, image))
            blocks.append([])
    
    if not len(blocks[-1]):
        blocks.pop()
    
    return blocks


def inference(video, predictions_per_second, video_output_path, precision, verbose):
    total = int(video.fps * video.duration)
    blocks = get_blocks(video, predictions_per_second)
    
    if video_output_path:
        writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), video.fps, tuple(video.size))
    
    json_outputs = []
    position = 1

    if verbose:
        print('FPS          :', video.fps)
        print('Video size   :', video.size)
        print('Total frames :', total)
    
    for block in blocks:
        mid_index, mid_image = block.pop(int(len(block)//2))
        mid_output_image, mid_landmarks_outputs, landmarks_dict = inference_on_mid_index(mid_image, precision)

        for img_index, image in block:
            output_image, _ = draw_landmarks_on_faces(image, mid_landmarks_outputs)
            if img_index < mid_index:
                if video_output_path:
                    writer.write(cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
                if (img_index + 1) == mid_index:
                    if video_output_path:
                        writer.write(cv2.cvtColor(mid_output_image, cv2.COLOR_RGB2BGR))
                    json_outputs.append({'position': position, 'time': secondsToStr(mid_index/video.fps), 'coordinates': landmarks_dict})
                    position += 1
            elif img_index > mid_index:
                if video_output_path:
                    writer.write(cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

        if not len(block):
            if video_output_path:
                writer.write(cv2.cvtColor(mid_output_image, cv2.COLOR_RGB2BGR))
            json_outputs.append({'position': position, 'time': secondsToStr(mid_index/video.fps), 'coordinates': landmarks_dict})
    
    if video_output_path:
        writer.release()

    return json_outputs





class DepthewiseSeperableConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super(DepthewiseSeperableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(input_channels, input_channels, kernel_size, groups = input_channels, bias = False, **kwargs)
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias = False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class EntryBlock(nn.Module):
    def __init__(self):
        super(EntryBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.conv3_residual = nn.Sequential(
            DepthewiseSeperableConv2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            DepthewiseSeperableConv2d(64, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride = 2, padding = 1),
        )

        self.conv3_direct = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride = 2),
            nn.BatchNorm2d(128),
        )

        self.conv4_residual = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthewiseSeperableConv2d(128, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            DepthewiseSeperableConv2d(128, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride = 2, padding = 1)
        )

        self.conv4_direct = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride = 2),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        residual = self.conv3_residual(x)
        direct = self.conv3_direct(x)
        x = residual + direct
        
        residual = self.conv4_residual(x)
        direct = self.conv4_direct(x)
        x = residual + direct

        return x

class MiddleBasicBlock(nn.Module):
    def __init__(self):
        super(MiddleBasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthewiseSeperableConv2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256)
        )
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthewiseSeperableConv2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256)
        )
        self.conv3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthewiseSeperableConv2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)

        return x + residual


class MiddleBlock(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()

        self.block = nn.Sequential(*[MiddleBasicBlock() for _ in range(num_blocks)])

    def forward(self, x):
        x = self.block(x)

        return x

class ExitBlock(nn.Module):
    def __init__(self):
        super(ExitBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthewiseSeperableConv2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            DepthewiseSeperableConv2d(256, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(3, stride = 2, padding = 1)
        )

        self.direct = nn.Sequential(
            nn.Conv2d(256, 512, 1, stride = 2),
            nn.BatchNorm2d(512)
        )

        self.conv = nn.Sequential(
            DepthewiseSeperableConv2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            DepthewiseSeperableConv2d(512, 1024, 3, padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )

        self.dropout = nn.Dropout(0.3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        direct = self.direct(x)
        residual = self.residual(x)
        x = direct + residual
        
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        return x

class XceptionNet(nn.Module):
    def __init__(self, num_middle_blocks = 6):
        super(XceptionNet, self).__init__()

        self.entry_block = EntryBlock()
        self.middel_block = MiddleBlock(num_middle_blocks)
        self.exit_block = ExitBlock()

        self.fc = nn.Linear(1024, 136)

    def forward(self, x):
        x = self.entry_block(x)
        x = self.middel_block(x)
        x = self.exit_block(x)

        x = x.view(x.size(0), -1)
        
        x = self.fc(x)

        return x
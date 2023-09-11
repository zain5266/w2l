import os
import cv2
import requests
from torch.utils.model_zoo import load_url

from ..core import FaceDetector

from .net_s3fd import s3fd
from .bbox import *
from .detect import *




class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=os.path.join(os.getcwd(), 's3fd/s3fd.pth'), verbose=False):
        super(SFDDetector, self).__init__(device, verbose)

        # Initialise the face detector
        if not os.path.isfile(path_to_detector):
             
            if not os.path.isdir(os.path.join(os.getcwd(), 's3fd')):
                os.makedirs(os.path.join(os.getcwd(), 's3fd'))
            model_url ='https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth'
            response = requests.get(model_url)
            if response.status_code == 200:
                print("Downloading s3fd model")
                with open(path_to_detector, "wb") as file:
                    file.write(response.content)
                model_weights = torch.load(path_to_detector)
            else:
                print("s3fd model is not found to download")
        else:
            model_weights = torch.load(path_to_detector)

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image, device=self.device)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    def detect_from_batch(self, images):
        bboxlists = batch_detect(self.face_detector, images, device=self.device)
        keeps = [nms(bboxlists[:, i, :], 0.3) for i in range(bboxlists.shape[1])]
        bboxlists = [bboxlists[keep, i, :] for i, keep in enumerate(keeps)]
        bboxlists = [[x for x in bboxlist if x[-1] > 0.5] for bboxlist in bboxlists]

        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0

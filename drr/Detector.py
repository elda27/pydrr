from pycuda import gpuarray
from functools import wraps
from . import KernelManager
from . import utils
import numpy as np

class Detector:
    def __init__(self, image_size, pixel_spacing, image=None, *, cpu=None):
        self.image = np.zeros(image_size, dtype=np.float32) if image is None else image
        self.image_size = image_size
        self.pixel_spacing = pixel_spacing
        self.cpu = cpu

    def to_cpu(self):
        #assert self.cpu is not None
        if self.is_cpu():
            return self
        return self.cpu

    def to_gpu(self):
        #assert self.cpu is None
        if self.is_gpu():
            return self
        image_size = KernelManager.Module.get_global(
            'd_image_size', 
            np.array(self.image_size, dtype=np.float32)
            )
        return Detector(image_size, self.pixel_spacing, gpuarray.to_gpu(self.image), cpu=self)

    def is_cpu(self):
        return self.cpu is None

    def is_gpu(self):
        return self.cpu is not None

    @staticmethod
    def make_detector_size(image_size, n_channels):
        return np.array((image_size[0], image_size[1], n_channels), dtype=np.int32)

from dataclasses import dataclass
from datetime import datetime, timedelta
import io
import os
import struct
import subprocess
from urllib import request
import warnings

import cv2
import imageio
import m3u8
import numpy as np

from mlworkflow import Dataset, AugmentedDataset, SideRunner, TransformedDataset


def gamma_correction(image, gammas=np.array([1.0, 1.0, 1.0])):
    image = image.astype(np.float32)
    image = image ** (1/gammas)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def crop_padded(array, x_slice, y_slice, padding):
    pad_width = [[padding]*2]*2 + (len(array.shape)-2)*[[0,0]]
    return np.pad(array, pad_width)[y_slice.start+padding:y_slice.stop+padding, x_slice.start+padding:x_slice.stop+padding]

try:
    from matplotlib import pyplot as plt
except ImportError:
    warnings.warn("Failed importing matplotlib")
else:
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle_rgb = list(map(lambda color: list(map(lambda c: int(c, 16), [color[1:3], color[3:5], color[5:7]])), color_cycle))

class DefaultDict(dict):
    def __init__(self, factory):
        self.factory = factory
    def __missing__(self, key):
        self[key] = self.factory(key)
        return self[key]


class LazyGeneratorBackedList(list):
    def __init__(self, gen):
        self.gen = gen
    def next(self):
        item = next(self.gen, None)
        if item is None:
            raise StopIteration
        self.append(item)
    def __getitem__(self, i):
        while i < -len(self) or len(self) <= i:
            self.next()
        return super().__getitem__(i)


class DefaultList(list):
    def __init__(self, *args, default=None, default_factory=None):
        super().__init__(*args)
        self.default_factory = default_factory or (lambda x: default)
    def __getitem__(self, i):
        if i < -len(self) or len(self) <= i:
            return self.default_factory(i)
        return super().__getitem__(i)


class DelayedCallback:
    def __init__(self, callback, timedelta=timedelta(seconds=10)):
        self.timedelta = timedelta
        self.last = datetime.now()
        self.callback = callback
    def __call__(self):
        now = datetime.now()
        if now - self.last > self.timedelta:
            self.last = now
            self.callback()
    def __del__(self):
        try:
            self.callback()
        except:
            pass

class VideoReaderDataset(Dataset):
    cap = None
    def __init__(self, filename, scale_factor=None, output_shape=None):
        raise
        # TODO: use instead
        # vid = imageio.get_reader("/home/gva/KS-FR-STCHAMOND_93815_concatenated.mp4",  'ffmpeg')
        # nums = [0, 1, 2]
        # for num in nums:
        #     image = vid.get_data(num)
        assert not scale_factor or not output_shape, "You cannot provide both 'scale_factor' and 'output_shape' arguments."
        self.cap = cv2.VideoCapture(filename)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        shape = tuple([int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))])
        if scale_factor:
            shape = tuple(int(x*scale_factor) for x in shape)
        elif output_shape:
            shape = output_shape
        self.shape = tuple(x-x%2 for x in shape) # make sure shape is even
    def __del__(self):
        if self.cap is not None:
            self.cap.release()
    def yield_keys(self):
        yield from range(self.frame_count)
    def query_item(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = self.cap.read()
        if frame is None:
            return None
        frame = cv2.resize(frame, self.shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

class M3u8PlaylistDataset(Dataset):
    def __init__(self, filename, download_folder=None):
        self.playlist = m3u8.load(filename)
        self.download_folder = download_folder
    def yield_keys(self):
        yield from self.playlist.segments
    def query_item(self, key):
        if self.download_folder is not None:
            filename = os.path.join(self.download_folder, os.path.basename(key.uri))
            request.urlretrieve(key.uri, filename)
            return filename
        return key.uri

class VideoFileNameToDatasetReaderTransform():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __call__(self, key, filename):
        return VideoReaderDataset(filename, **self.kwargs)

class VideoFromPlaylistDataset(AugmentedDataset):
    def augment(self, root_key, dataset):
        for key in dataset.yield_keys():
            item = dataset.query_item(key)
            if item is not None:
                yield (root_key, root_key.uri, key), item

def VideoDataset(filename, **kwargs):
    folder = os.path.dirname(filename)
    supported_formats = {
        ".m3u8": lambda name: VideoFromPlaylistDataset(
            TransformedDataset(
                M3u8PlaylistDataset(name, download_folder=folder),
                [VideoFileNameToDatasetReaderTransform(**kwargs)]
            )
        ),
        ".mp4": lambda name: VideoReaderDataset(name, **kwargs) # pylint: disable=unnecessary-lambda
    }
    return supported_formats[os.path.splitext(filename)[1]](filename)


def concatenate_chunks(output_filename, *chunk_urls):
    side_runner = SideRunner(10)
    for chunk_url in chunk_urls:
        side_runner.run_async(subprocess.run, ["wget", chunk_url])
    side_runner.collect_runs()

    command = [
        'ffmpeg',
        '-y',
        '-protocol_whitelist "concat,file,http,https,tcp,tls"',
        '-i "concat:{}"'.format("|".join([url[url.rfind("/")+1:] for url in chunk_urls])),
        '-c:a copy',
        '-c:v copy',
        '-movflags faststart',
        output_filename
    ]
    os.system(" ".join(command))
    #subprocess.run(command) # For obscure reason, subprocess doesn't work here

@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int
    @property
    def x_slice(self):
        return slice(int(self.x), int(self.x+self.w), None)
    @property
    def y_slice(self):
        return slice(int(self.y), int(self.y+self.h), None)

    def increase_box(self, max_width, max_height, aspect_ratio=None, margin=0, padding=0):
        """ Adapt the bounding-box s.t. it
                - is increased by `margin` on all directions
                - lies within the source image of size `max_width`x`max_height`
                - has the aspect ratio given by `aspect_ratio` (if not None)
                - contains the original bounding-box (box is increased if necessary, up to source image limits)
            Arguments:
                max_width (int)      - width of input image
                max_height (int)     - height of input image
                aspect_ratio (float) - output aspect-ratio
                margin (int)         - margin in pixels to be added on 4 sides
            Returns:
                x_slice (slice) - the horizontal slice
                y_slice (slice) - the vertical slice
        """
        top   = max(-padding,           int(self.y-margin))
        bot   = min(max_height+padding, int(self.y+self.h+margin))
        left  = max(-padding,           int(self.x-margin))
        right = min(max_width+padding,  int(self.x+self.w+margin))

        if aspect_ratio is None:
            return slice(left, right, None), slice(top, bot, None)

        if padding:
            try:
                raise NotImplementedError("increase_box method doesn't support padding when aspect ratio is given")
            except:
                pass

        w = right - left
        h = bot - top
        if w/h > aspect_ratio: # box is wider
            h = int(w/aspect_ratio)
            if h > max_height: # box is too wide
                h = max_height
                w = int(max_height*aspect_ratio)
                left = max_width//2 - w//2
                return slice(left, w, None), slice(0, h, None)
            cy = (bot+top)//2
            if cy + h//2 > max_height: # box hits the top
                return slice(left, right, None), slice(0, h, None)
            if cy - h//2 < 0: # box hits the bot
                return slice(left, right, None), slice(max_height-h, max_height, None)
            return slice(left, right, None), slice(cy-h//2, cy-h//2+h, None)

        if w/h < aspect_ratio: # box is taller
            w = int(h*aspect_ratio)
            if w > max_width: # box is too tall
                w = max_width
                h = int(max_width/aspect_ratio)
                top = max_height//2 - h//2
                return slice(0, w, None), slice(top, top+h, None)
            cx = (left+right)//2
            if cx + w//2 > max_width: # box hits the right
                return slice(max_width-w, max_width, None), slice(top, bot, None)
            if cx - w//2 < 0: # box hits the left
                return slice(0, w, None), slice(top, bot, None)
            return slice(cx-w//2, cx-w//2+w, None), slice(top, bot, None)

        # else: good aspect_ratio
        return slice(left, right, None), slice(top, bot, None)


class VideoMaker:
    def __init__(self, filename="output.mp4", framerate=15):
        self.filename = filename
        self.writer = imageio.get_writer(filename, fps=framerate)
    def __enter__(self):
        return self
    def __call__(self, image):
        self.writer.append_data(image)
    def __exit__(self, exc_type, exc_value, traceback):
        if self.writer is not None:
            self.writer.close()

class VideoMaker_obsolete():
    format_map = {
        ".mp4": 'mp4v',
        ".avi": 'XVID',
        ".mpeg4": 'H264'
    }
    writer = None
    def __init__(self, filename="output.mp4", framerate=15):
        self.filename = filename
        self.framerate = framerate
        self.fourcc = cv2.VideoWriter_fourcc(*self.format_map[os.path.splitext(filename)[1]])
    def __enter__(self):
        return self
    def __call__(self, image):
        if self.writer is None:
            shape = (image.shape[1], image.shape[0])
            self.writer = cv2.VideoWriter(filename=self.filename, fourcc=self.fourcc, fps=self.framerate, frameSize=shape, apiPreference=cv2.CAP_FFMPEG)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.writer.write(image)
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.release()
            self.writer = None
            print("{} successfully written".format(self.filename))
    def __del__(self):
        if self.writer:
            self.writer.release()
            self.writer = None
            print("{} successfully written".format(self.filename))


def blend(image, saliency, alpha=1.0, beta=0.5, gamma=0.0):
    #assert image.dtype == np.uint8 and image.shape[2] == 3
    #assert saliency.dtype == np.uint8

    if len(saliency.shape) == 2 or saliency.shape[2] == 1:
        saliency = np.dstack((saliency, saliency, saliency))
    return cv2.addWeighted(image, alpha, saliency, beta, gamma)

# Image is 2D numpy array, q is quality 0-100
def jpegBlur(im, q):
    buf = io.BytesIO()
    imageio.imwrite(buf,im,format='jpg',quality=q)
    s = buf.getbuffer()
    return imageio.imread(s,format='jpg')

def setdefaultattr(obj, name, value):
    if not hasattr(obj, name):
        setattr(obj, name, value)
    return getattr(obj, name)


class MJPEGReader:
    def __init__(self, filename):
        self.fd = open(f"{filename}.idx", "rb")
        self.cap = cv2.VideoCapture(filename)
        self.header, self.version = struct.unpack("QI", self.fd.read(12))
    def __del__(self):
        if self.cap:
            self.cap.release()
    def __iter__(self):
        return self
    def __next__(self):
        try:
            tvsec, tvusec, offset, frame_idx, other = struct.unpack("IIQII", self.fd.read(24))
        except:
            raise StopIteration
        found, image = self.cap.read()
        timestamp = round(tvsec*1000+tvusec/1000)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
        return timestamp, offset, frame_idx, other, image

def colorify_heatmap(heatmap, colormap="jet"):
    return (plt.get_cmap(colormap)(heatmap)[...,0:3]*255).astype(np.uint8)

"""
Provides functions for processing a video file or collage into numpy arrays
for RGB data and optical flow data, which can be used with the I3D model.
"""

import cv2
import numpy as np
import traceback
from random import randint


class VideoClip:
    def __init__(
        self,
        input,
        image_size,
        nframes,
        flip=False,
        flow_crop=True,
        time_stretch=True,
        randomize_timecrop=False,
    ):
        self.input = input
        self.image_size = image_size
        self.nframes = nframes
        self.flow_crop = flow_crop
        self.flip = flip
        self.time_stretch = time_stretch
        self.randomize_timecrop = randomize_timecrop
        self.start = None

    def _raw_numpy_array(
        self, nframes=None, fliplr=False, time_stretch=True, randomize_timecrop=False
    ):
        """
        Loads a video from the given file. Will set the number
        of frames to `nframes` if this parameter is not `None`.

        Returns:
        - (width, height, arr): The width and height of the video,
            and a numpy array with the parsed contents of the video.
        """
        if isinstance(self.input, str):
            # Read video
            from_file = True
            video_file = self.input
            cap = cv2.VideoCapture(video_file)

            # Get properties of the video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            from_file = False
            kolaz = self.input
            # Get properties of the video
            frame_count = kolaz.num_frames
            w = kolaz.w
            h = kolaz.h
        # Min allowed height or width (whatever is smaller), in pixels
        min_dimension = 256.0

        # Determine scaling factors of width and height
        assert min(w, h) > 0, "Cannot resize {} with W={}, H={}".format(
            self.input, w, h
        )
        scale = min_dimension / min(w, h)
        w = int(w * scale)
        h = int(h * scale)

        buf = np.zeros((1, frame_count, h, w, 3), np.dtype("float32"))
        fc, flag = 0, True
        if from_file:
            while fc < frame_count and flag:
                flag, image = cap.read()

                if flag:
                    image = cv2.resize(image, (w, h))
                    if fliplr:
                        image = cv2.flip(image, 1)
                    buf[0, fc] = image

                fc += 1

            cap.release()
        else:
            for i in range(frame_count):
                image = kolaz.video_buffer[:, :, :, i]
                buf[0, i] = cv2.resize(image, (w, h))

        if nframes is not None:
            if nframes < frame_count:
                if time_stretch:
                    newtimes = np.linspace(0, frame_count - 1, nframes, dtype=np.int)
                else:
                    xtra = frame_count - nframes
                    if randomize_timecrop:
                        if self.start is None:
                            self.start = randint(0, xtra - 1)
                    else:
                        self.start = int(xtra / 2)
                        stop = self.start + nframes
                        newtimes = np.arange(self.start, stop)
                buf = buf[:, newtimes, :, :, :]
            else:
                buf = np.resize(buf, (1, nframes, h, w, 3))

        return w, h, buf

    def _scaled_raw_numpy_array(
        self,
        ww,
        wh,
        nframes=None,
        fliplr=False,
        time_stretch=True,
        randomize_timecrop=False,
    ):
        """
        Loads a video from the given file. Will set the number
        of frames to `nframes` if this parameter is not `None`.
        Resize frames to ww,wh

        Returns:
        - (width, height, arr): The width and height of the video,
            and a numpy array with the parsed contents of the video.
        """

        # Read video
        if isinstance(self.input, str):
            video_file = self.input
            from_file = True
        else:
            kolaz = self.input
            from_file = False

        if from_file:
            cap = cv2.VideoCapture(video_file)
            # Get properties of the video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            try:
                assert min(w, h) > 0, "Cannot resize {} with W={}, H={}".format(
                    video_file, w, h
                )
            except:
                print(video_file)
                traceback.print_exc()
        else:
            frame_count = kolaz.num_frames
            w = kolaz.w
            h = kolaz.h
            assert min(w, h) > 0, "Cannot resize {} with W={}, H={}".format(kolaz, w, h)

        buf = np.zeros((1, frame_count, wh, ww, 3), np.dtype("float32"))
        fc, flag = 0, True

        if from_file:
            while fc < frame_count and flag:
                flag, image = cap.read()

                if flag:
                    image = cv2.resize(image, (ww, wh))
                    if fliplr:
                        image = cv2.flip(image, 1)
                    buf[0, fc] = image

                fc += 1

            cap.release()
        else:
            for i in range(frame_count):
                image = kolaz.video_buffer[:, :, :, i]
                buf[0, i] = cv2.resize(image, (ww, wh))

        if nframes is not None:
            if nframes < frame_count:
                if time_stretch:
                    newtimes = np.linspace(0, frame_count - 1, nframes, dtype=np.int)
                else:
                    xtra = frame_count - nframes
                    if randomize_timecrop:
                        if self.start is None:
                            self.start = randint(0, xtra - 1)
                    else:
                        self.start = int(xtra / 2)
                        stop = self.start + nframes
                        newtimes = np.arange(self.start, stop)
                buf = buf[:, newtimes, :, :, :]
            else:
                buf = np.resize(buf, (1, nframes, wh, ww, 3))

        return ww, wh, buf

    def _crop_video(self, numpy_video, size, desired_size):
        """
        Crop a video of the given size (WIDTH, HEIGHT) into a square of `desired_size`.
        The video is represented as a numpy array. This func is for internal usage.
        """

        w, h = size
        h1, h2 = int(h / 2) - int(desired_size / 2), int(h / 2) + int(desired_size / 2)
        w1, w2 = int(w / 2) - int(desired_size / 2), int(w / 2) + int(desired_size / 2)
        return numpy_video[:, :, h1:h2, w1:w2, :]


    def rgb_data_uncropped(
        self,
        size=None,
        nframes=None,
        flip=None,
        time_stretch=None,
        randomize_timecrop=None,
    ):
        """
        Loads a numpy array of shape (1, nframes, size, size, 3) from a video file.
        Values contained in the array are based on RGB values of each frame in the video.

        Parameter `size` should be an int (pixels) for a square cropping of the video.
        Omitting the parameter `nframes` will preserve the original # frames in the video.
        time_stretch: uzima se svaki n-ti frame iz cijele sekvence, inače se uzimaju susjedni frameovi
        """
        if size is None:
            size = self.image_size
        if nframes is None:
            nframes = self.nframes
        if flip is None:
            flip = self.flip
        if time_stretch is None:
            time_stretch = self.time_stretch
        if randomize_timecrop is None:
            randomize_timecrop = self.randomize_timecrop

        # Load video into numpy array
        w, h, buf = self._scaled_raw_numpy_array(
            size,
            size,
            nframes=nframes,
            fliplr=flip,
            time_stretch=time_stretch,
            randomize_timecrop=randomize_timecrop,
        )
        # Scale pixels between -1 and 1
        buf[0, :] = ((buf[0, :] / 255.0) * 2) - 1

        # Select center crop from the video
        return buf

    def flow_data(
        self,
        size=None,
        nframes=None,
        flip=None,
        time_stretch=None,
        randomize_timecrop=None,
        flow_crop=None,
    ):
        """
        Loads a numpy array of shape (1, nframes, size, size, 2) from a video file.
        Values contained in the array are based on optical flow of the video.
        https://docs.opencv.org/3.1.0/d6/d39/classcv_1_1cuda_1_1OpticalFlowDual__TVL1.html

        Parameter `size` should be an integer (pixels) for a square cropping of the video.
        Omitting the parameter `nframes` will preserve the original # frames in the video.
        """
        if size is None:
            size = self.image_size
        if nframes is None:
            nframes = self.nframes
        if flip is None:
            flip = self.flip
        if time_stretch is None:
            time_stretch = self.time_stretch
        if randomize_timecrop is None:
            randomize_timecrop = self.randomize_timecrop
        if flow_crop is None:
            flow_crop = self.flow_crop
        # Load video into numpy array, and crop the video
        if flow_crop:
            w, h, buf = self._raw_numpy_array(
                nframes=nframes,
                fliplr=flip,
                time_stretch=time_stretch,
                randomize_timecrop=randomize_timecrop,
            )
            buf = self._crop_video(buf, (w, h), size)
        else:
            w, h, buf = self._scaled_raw_numpy_array(
                size,
                size,
                nframes=nframes,
                fliplr=flip,
                time_stretch=time_stretch,
                randomize_timecrop=randomize_timecrop,
            )

        num_frames = buf.shape[1]
        flow = np.zeros((1, num_frames, size, size, 2), dtype="float32")

        # Convert to grayscale
        buf = np.dot(buf, np.array([0.2989, 0.5870, 0.1140]))

        # Apply optical flow algorithm
        for i in range(1, num_frames):
            prev, cur = buf[0, i - 1], buf[0, i]
            cur_flow = cv2.calcOpticalFlowFarneback(
                prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Truncate values to [-20, 20] and scale from [-1, 1]
            cur_flow[cur_flow < -20] = -20
            cur_flow[cur_flow > 20] = 20
            cur_flow /= 20
            flow[0, i] = cur_flow

        return flow

__author__ = 'tbeltramelli'

from UMedia import *
from Filtering import *

class PersonTracker:
    _result = None
    _data = None
    _counter = 0

    def __init__(self, video_path, map_path, tracking_data_path):
        self._data = self._get_tracking_data(tracking_data_path)

        UMedia.load_media(video_path, self.process)

    def process(self, img):
        self._result = img
        img = Filtering.apply_box_filter(Filtering.get_gray_scale_image(img), 3)

        self._counter += 1

        row = self._data[self._counter]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for i, box in enumerate(row):
            cv2.rectangle(self._result, box[0], box[1], colors[i])

        UMedia.show(self._result)

    def _get_tracking_data(self, tracking_data_path):
        data = np.loadtxt(tracking_data_path)
        length, n = data.shape

        boxes = []
        for i in range(length):
            boxes.append(self._get_tracking_box(data[i, :]))

        return boxes

    def _get_tracking_box(self, data):
        points = [(int(data[i]), int(data[i + 1])) for i in range(0, len(data) - 1, 2)]
        boxes = []

        for i in range(0, len(data) / 2, 2):
            box = tuple(points[i:i + 2])
            boxes.append(box)
        return boxes

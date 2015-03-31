__author__ = 'tbeltramelli'

from UMedia import *
from AHomography import *


class PersonTracker(AHomography):
    _input = None
    _data = None
    _map = None
    _counter = 0
    _output_path = ""

    def __init__(self, video_path, map_path, tracking_data_path, output_path):
        self._data = self.get_tracking_data(tracking_data_path)
        self._map = UMedia.get_image(map_path)
        self._output_path = output_path

        UMedia.load_media(video_path, self.process)

    def process(self, img):
        self._input = img

        self.build_homography([img, self._map])

        x, y = self.get_person_position()
        x, y = self.get_2d_transform(x, y)

        cv2.circle(self._map, (x, y), 3, (0, 255, 0))

        UMedia.show(self._input, self._map)

    def get_person_position(self):
        self._counter += 1

        if self._counter >= len(self._data):
            cv2.imwrite(self._output_path, self._map)
            return 0, 0

        row = self._data[self._counter]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for i, box in enumerate(row):
            cv2.rectangle(self._input, box[0], box[1], colors[i])

        box = row[len(row)-1]
        base_x = box[0][0] + ((box[1][0] - box[0][0]) / 2)
        base_y = box[1][1]

        cv2.circle(self._input, (base_x, base_y), 1,  (0, 255, 255))

        return base_x, base_y

    def get_tracking_data(self, tracking_data_path):
        data = np.loadtxt(tracking_data_path)
        length, n = data.shape

        boxes = []
        for i in range(length):
            boxes.append(self.get_tracking_boxes(data[i, :]))

        return boxes

    def get_tracking_boxes(self, data):
        points = [(int(data[i]), int(data[i + 1])) for i in range(0, len(data) - 1, 2)]
        boxes = []

        for i in range(0, len(data) / 2, 2):
            box = tuple(points[i:i + 2])
            boxes.append(box)
        return boxes

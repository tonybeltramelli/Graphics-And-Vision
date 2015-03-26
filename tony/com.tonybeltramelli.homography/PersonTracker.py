__author__ = 'tbeltramelli'

from UMedia import *
from Filtering import *
from UMath import *

class PersonTracker:
    _input = None
    _data = None
    _map = None
    _homography = None
    _counter = 0

    def __init__(self, video_path, map_path, tracking_data_path):
        self._data = self._get_tracking_data(tracking_data_path)
        self._map = UMedia.get_image(map_path)

        UMedia.load_media(video_path, self.process)

    def process(self, img):
        self._input = img

        if self._homography is None:
            self._homography = self._get_homography_from_mouse([img, self._map])

        x, y = self._get_person_position()
        x, y = UMath.get_2D_transform_from_homography(x, y, self._homography)

        cv2.circle(self._map, (x, y), 5, (0, 255, 0))

        UMedia.show(self._map)

    def _get_person_position(self):
        self._counter += 1

        if self._counter >= len(self._data):
            return 0, 0

        row = self._data[self._counter]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for i, box in enumerate(row):
            cv2.rectangle(self._input, box[0], box[1], colors[i])

        box = row[len(row)-1]
        base_x = box[0][0] + ((box[1][0] - box[0][0]) / 2)
        base_y = box[1][1]

        cv2.circle(self._input, (base_x, base_y), 1,  (0, 255, 255))

        #UMedia.show(self._input)

        return base_x, base_y

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

    def _get_homography_from_mouse(self, images, n=4):
        image_points = []
        fig = figure(1)

        for i, img in enumerate(images):
            img = copy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            ax = subplot(1, 2, i + 1)
            ax.imshow(img)

            title("Select " + str(n) + " points.")

            fig.canvas.draw()
            ax.hold('On')

            image_points.append(fig.ginput(n, -1))

            for p in image_points[i]:
                cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), 2)

            ax.imshow(img)

        close(fig)

        points1 = np.array([[x, y] for (x, y) in image_points[0]])
        points2 = np.array([[x, y] for (x, y) in image_points[len(image_points) - 1]])

        homography, mask = cv2.findHomography(points1, points2)
        return homography

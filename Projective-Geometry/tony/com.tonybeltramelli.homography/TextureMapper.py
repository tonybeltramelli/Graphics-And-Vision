__author__ = 'tbeltramelli'

from AHomography import *
from Filtering import *
from UInteractive import *


class TextureMapper(AHomography):
    _result = None
    _texture = None
    _map = None
    _texture_position = None

    def __init__(self, homography_output_path):
        self._homography_output_path = homography_output_path

    def map(self, video_path, texture_path, is_automatic):
        self._texture = UMedia.get_image(texture_path)
        self._homography = None

        if is_automatic:
            UMedia.load_media(video_path, self.process_with_chessboard)
        else:
            UMedia.load_media(video_path, self.process_with_homography)

    def process_with_homography(self, img):
        if self._homography is None:
            self._homography = self.get_homography_part_from_mouse(self._texture, img)

        self._result = self.get_mapped_texture(self._texture, img, self._homography)

        UMedia.show(self._result)

    def process_with_chessboard(self, img, pattern_size=(9, 6), to_draw=False):
        self._result = cv2.pyrDown(img)

        pos = [0, pattern_size[0] - 1, pattern_size[0] * (pattern_size[1] - 1), (pattern_size[0] * pattern_size[1]) - 1]
        corners = []

        img = Filtering.get_gray_scale_image(self._result)
        is_found, coordinates = cv2.findChessboardCorners(img, pattern_size)

        if is_found:
            term = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1
            cv2.cornerSubPix(img, coordinates, (5, 5), (-1, -1), term)
            if to_draw:
                cv2.drawChessboardCorners(self._result, pattern_size, coordinates, is_found)

            for p in pos:
                corner = int(coordinates[p, 0, 0]), int(coordinates[p, 0, 1])
                if to_draw:
                    cv2.circle(self._result, corner, 10, (255, 0, 0))
                corners.append(corner)

            self._homography = self.get_homography_from_coordinates(self._texture, corners)
            self._result = self.get_mapped_texture(self._texture, self._result, self._homography)

        UMedia.show(self._result)

    def map_realistically(self, video_path, map_path, texture_path):
        self._map = UMedia.get_image(map_path)
        self._texture = UMedia.get_image(texture_path)
        self._homography = None

        UMedia.load_media(video_path, self.process_realistically)

    def process_realistically(self, img):
        self._result = img

        if self._homography is None:
            self.define_map_homography([img, self._map])
            self._homography = np.linalg.inv(self._homography)

        x, y = self.get_texture_position(self._map)
        x, y = self.get_2d_transform_from_homography(x, y, self._homography)

        self.apply_texture(0.5, x, y, self._homography)

        UMedia.show(self._result)

    def apply_texture(self, scale, x, y, homography):
        height, width, layers = self._texture.shape

        w = width * scale
        h = height * scale

        p1 = self.get_2d_transform_from_homography(x - (w/2), y - (h/2), homography)
        p2 = self.get_2d_transform_from_homography(x + (w/2), y - (h/2), homography)
        p3 = self.get_2d_transform_from_homography(x - (w/2), y + (h/2), homography)
        p4 = self.get_2d_transform_from_homography(x + (w/2), y + (h/2), homography)

        corners = [p1, p2, p3, p4]

        h = self.get_homography_from_coordinates(self._texture, corners)
        self._result = self.get_mapped_texture(self._texture, self._result, h)

    def get_texture_position(self, img):
        if self._texture_position is None:
            self._texture_position = UInteractive.select_points_in_images([img], 1)[0][0]

        return self._texture_position


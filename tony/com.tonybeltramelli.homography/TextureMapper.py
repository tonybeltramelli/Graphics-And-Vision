__author__ = 'tbeltramelli'

from UMedia import *
from AHomography import *
from Filtering import *

class TextureMapper(AHomography):
    _result = None
    _texture = None

    def map(self, video_path, texture_path, is_automatic):
        self._texture = UMedia.get_image(texture_path)

        if is_automatic:
            UMedia.load_media(video_path, self.process_with_chessboard)
        else:
            UMedia.load_media(video_path, self.process_with_homography)

    def process_with_homography(self, img):
        self.build_homography([self._texture, img])
        self._result = self.get_mapped_texture(self._texture, img)

        UMedia.show(self._result)

    def process_with_chessboard(self, img, pattern_size=(9, 6), to_draw=False):
        self._result = cv2.pyrDown(img)

        pos = [0, pattern_size[0] - 1, pattern_size[0] * (pattern_size[1] - 1), (pattern_size[0] * pattern_size[1]) - 1]
        corners = []

        img = Filtering.get_gray_scale_image(self._result)
        found, coordinates = cv2.findChessboardCorners(img, pattern_size)

        if found:
            term = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1
            cv2.cornerSubPix(img, coordinates, (5, 5), (-1, -1), term)
            if to_draw:
                cv2.drawChessboardCorners(self._result, pattern_size, coordinates, found)

            for p in pos:
                corner = int(coordinates[p, 0, 0]), int(coordinates[p, 0, 1])
                if to_draw:
                    cv2.circle(self._result, corner, 10, (255, 0, 0))
                corners.append(corner)

            self.build_homography_from_coordinates(self._texture, corners)
            self._result = self.get_mapped_texture(self._texture, self._result)

        UMedia.show(self._result)



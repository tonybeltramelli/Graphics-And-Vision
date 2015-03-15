__author__ = 'tbeltramelli'

class UMath:

    @staticmethod
    def normalize(range_min, range_max, x, x_min, x_max):
        return range_min + (((x - x_min) * (range_max - range_min)) / (x_max - x_min))

    @staticmethod
    def is_in_area(x, y, width, height):
        return ((x > width/4) and (x < 3 * (width/4))) and ((y > height/4) and (y < 3 * (height/4)))
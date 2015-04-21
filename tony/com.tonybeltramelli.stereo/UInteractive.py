__author__ = 'tbeltramelli'

from UMedia import *


class UInteractive:

    @staticmethod
    def select_points_in_images(images, n=4):
        image_points = []
        fig = figure(1)

        for i, img in enumerate(images):
            img = copy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            ax = subplot(1, len(images), i + 1)
            ax.imshow(img)

            title("Select " + str(n) + " points.")

            fig.canvas.draw()
            ax.hold('On')

            image_points.append(fig.ginput(n, -1))

            for p in image_points[i]:
                cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), 2)

            ax.imshow(img)

        close(fig)

        return image_points
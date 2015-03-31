__author__ = 'tbeltramelli'

from PersonTracker import *
from TextureMapper import *

#pt = PersonTracker("../data/media/SunClipDS.avi", "../data/media/ITUMap.bmp", "../data/media/trackingdata.dat", "../data/output/map.jpg")

tm = TextureMapper()
#tm.map("../data/media/SunClipDS.avi", "../data/media/ITULogo.jpg", False)
#tm.map("../data/media/grid1.mp4", "../data/media/ITULogo.jpg", True)
tm.realistic_map("../data/media/SunClipDS.avi", "../data/media/ITUMap.bmp", "../data/media/ITULogo.jpg")

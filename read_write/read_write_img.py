import cv2
from utils.utils import (
    get_files_of_type,
    load_images_of_type,
    get_data_dir,
    write_dicom_to_jpg,
    write_dicom_to_png
)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

current_director = Path.cwd() / "read_write"
dataDir = get_data_dir()
png_jpg_dir = dataDir / "Practice_PNGandJPG"
dicom_dir = dataDir / "Practice_DICOM"

print(dataDir)

pngFiles = get_files_of_type(png_jpg_dir, 'png')
jpgFiles = get_files_of_type(png_jpg_dir, 'jpg')
dcmFiles = get_files_of_type(dataDir, 'dcm')

loadPng = load_images_of_type(pngFiles, 'png')
loadJpg = load_images_of_type(jpgFiles, 'jpg')
loadDcm = load_images_of_type(dcmFiles, 'dcm')


# Show images
horizontal_combined = np.hstack((loadPng[0], loadJpg[0]))
resized = cv2.resize(horizontal_combined, (1000, 500))
cv2.imshow("show", resized)

plt.imshow(loadDcm[0], cmap="gray")
plt.show()

# writing Images
cv2.imwrite(current_director / "write.png",loadPng[0])
cv2.imwrite(current_director / "write.jpg",loadJpg[0])

#write dicom to png and jpg
write_dicom_to_png(loadDcm[0], current_director, "writeDicom")
write_dicom_to_jpg(loadDcm[0], current_director, "writeDicom")

# close windows when done
cv2.waitKey(0)
cv2.destroyAllWindows()
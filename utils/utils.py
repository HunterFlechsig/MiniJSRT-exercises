from pathlib import Path
import cv2
import pydicom

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def get_data_dir() -> Path: return get_project_root() / "data"

def get_output_dir() -> Path: return get_project_root() / "output"

def get_current_dir() -> Path: return Path.cwd();

def get_files_of_type(directory, type):
    print(f"Searching in: {directory}")
    return list(directory.rglob(f"*.{type}"))

def load_dicom_file(file):
    ds = pydicom.dcmread(file)
    img = ds.pixel_array
    return img

def load_images_of_type(files, type):
    loaded_files = []
    for file in files:
        if(type == 'dcm'):
            loaded_files.append(load_dicom_file(file))
        else:
            loaded_files.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
    return loaded_files

def get_and_load_files_of_type(directory, file_ext):
    files = get_files_of_type(directory, file_ext)
    return load_images_of_type(files, file_ext)

def dicomeTo8bit(image):
    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype("uint8")
    return img

def write_dicom_to_png(image, dir, name):
    img = dicomeTo8bit(image)
    cv2.imwrite(dir / f"{name}.png", img)

def write_dicom_to_jpg(image, dir, name):
    img = dicomeTo8bit(image)
    cv2.imwrite(dir / f"{name}.jpg", img)
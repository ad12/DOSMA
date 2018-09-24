import h5py
import pickle
import os
import pandas as pd
import SimpleITK as sitk

DATA_EXT = 'data'
INFO_EXT = 'info'


def get_metadata_dirpath(dirpath):
    return check_dir(os.path.join(dirpath, 'metadata'))


def get_visualization_dirpath(dirpath):
    return check_dir(os.path.join(dirpath, 'visualization'))


def get_subdirs(dirpath):
    if not os.path.isdir(dirpath):
        raise NotADirectoryError('%s not a directory' % dirpath)

    subdirs =[]
    for file in os.listdir(dirpath):
        possible_dir = os.path.join(dirpath, file)
        if os.path.isdir(possible_dir):
            subdirs.append(file)

    return subdirs


def check_dir(dir_path):
    """
    If directory does not exist, make directory
    :param dir_path: path to directory
    :return: path to directory
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


def save_pik(filepath, data):
    """
    Save data using pickle
    :param data: data to save
    :param filepath: a string
    :return:
    """
    check_dir(os.path.dirname(filepath))
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pik(filepath):
    """
    Load data using pickle
    :param filepath: filepath to load from
    :return: data saved using save_pik
    """
    if (not os.path.isfile(filepath)):
        raise FileNotFoundError('%s does not exist' % filepath)

    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_h5(filepath, data_dict):
    check_dir(os.path.dirname(filepath))
    with h5py.File(filepath, 'w') as f:
        for key in data_dict.keys():
            f.create_dataset(str(key), data=data_dict[key])


def load_h5(filepath):
    if (not os.path.isfile(filepath)):
        raise FileNotFoundError('%s does not exist' % filepath)

    data = dict()
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            data[key] = f.get(key)


def save_tables(filepath, data_frames, sheet_names=None):
    check_dir(os.path.dirname(filepath))
    writer = pd.ExcelWriter(filepath)

    if sheet_names is None:
        sheet_names = []
        for i in range(len(data_frames)):
            sheet_names.append('Sheet%d' % (i+1))

    if len(data_frames) != len(sheet_names):
        raise ValueError('Number of data_frames and sheet_frames should be the same')

    for i in range(len(data_frames)):
        df = data_frames[i]
        df.to_excel(writer, sheet_names[i])

    writer.save()


def save_nifti(filepath, img_array, dicom_header=None):
    check_dir(os.path.dirname(filepath))
    image = sitk.GetImageFromArray(img_array)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(filepath)
    writer.Execute(image)


def load_nifti(filepath):
    reader = sitk.ImageFileReader()
    reader.SetFileName(filepath)
    image = reader.Execute()

    return sitk.GetArrayFromImage(image)


if __name__ == '__main__':
    from scan_sequences.cube_quant import CubeQuant
    cq = CubeQuant('../dicoms/healthy07/008', 'dcm')
    img_arr = cq.volume
    save_nifti('test.nii', img_arr)
    img_loaded = load_nifti('test.nii')

    print(img_arr.shape)
    print(img_loaded.shape)

    assert (img_arr == img_loaded).all()


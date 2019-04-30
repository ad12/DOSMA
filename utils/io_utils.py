import os
import pickle

import h5py
import pandas as pd

__all__ = ['check_dir', 'save_pik', 'load_pik', 'save_h5', 'load_h5', 'save_tables']


def check_dir(dir_path):
    """Make directory is directory does not exist
    :param dir_path: path to directory
    :return: path to directory
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    return dir_path


def save_pik(filepath, data):
    """Save data using pickle
    :param data: data to save
    :param filepath: a string
    """
    check_dir(os.path.dirname(filepath))
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pik(filepath):
    """Load data using pickle
    :param filepath: filepath to load from
    :return: data saved using save_pik
    """
    if (not os.path.isfile(filepath)):
        raise FileNotFoundError('%s does not exist' % filepath)

    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_h5(filepath, data_dict):
    """Save data in H5DF format
    :param filepath: path to h5 file to create
    :param data_dict: dictionary of data to store
    :return:
    """
    check_dir(os.path.dirname(filepath))
    with h5py.File(filepath, 'w') as f:
        for key in data_dict.keys():
            f.create_dataset(key, data=data_dict[key])


def load_h5(filepath):
    """Load data in H5DF format
    :param filepath: path to h5 file
    :return: dictionary of data values stored using save_h5
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError('%s does not exist' % filepath)

    data = dict()
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            data[key] = f.get(key).value

    return data


def save_tables(filepath, data_frames, sheet_names=None):
    """Save data in excel tables
    :param filepath: filepath to excel file
    :param data_frames: panda dataframes to use
    :param sheet_names: names of different sheets if storing multi-sheet data
    :return:
    """
    check_dir(os.path.dirname(filepath))
    writer = pd.ExcelWriter(filepath)

    if sheet_names is None:
        sheet_names = []
        for i in range(len(data_frames)):
            sheet_names.append('Sheet%d' % (i + 1))

    if len(data_frames) != len(sheet_names):
        raise ValueError('Number of data_frames and sheet_frames should be the same')

    for i in range(len(data_frames)):
        df = data_frames[i]
        df.to_excel(writer, sheet_names[i], index=False)

    writer.save()

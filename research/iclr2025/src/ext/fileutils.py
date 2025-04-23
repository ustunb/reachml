from pathlib import Path
import dill
import gzip
import pandas as pd

def save(obj, path, overwrite=False, check_save=False, mkdir = True, compress=False):
    """
    saves data as a pickle file on disk
    :param obj: object to save to disk
    :param path: path to create
    :return: saved path
    """
    f = Path(path)
    if f.is_file() and overwrite is False:
        raise IOError(f'file: {f} exists')

    if not f.parent.exists() and mkdir:
        f.parent.mkdir(parents = True, exist_ok = True)

    if compress:
        with gzip.open(f, 'wb') as outfile:
            dill.dump({'data': obj}, outfile, protocol=dill.HIGHEST_PROTOCOL)
    else:
        with open(f, 'wb') as outfile:
            dill.dump({'data': obj}, outfile, protocol=dill.HIGHEST_PROTOCOL)

    if check_save:
        loaded_obj = load(f, decompress=compress)
        if isinstance(loaded_obj, pd.DataFrame):
            assert loaded_obj.equals(obj)
        else:
            assert obj == loaded_obj
    print(f'saved to: {f}')
    return f

def load(path, decompress=False):
    """
    loads pickle file from disk
    :param path: path of the file
    :return: contents of file under 'data'
    """
    f = Path(path)
    if not f.is_file():
        raise IOError(f'file: {f} not found')

    if decompress:
        with gzip.open(f, 'rb') as infile:
            file_contents = dill.load(infile)
    else:
        with open(f, 'rb') as infile:
            file_contents = dill.load(infile)

    assert 'data' in file_contents, f'contents of {f} is missing a field called `data`'
    obj = file_contents['data']
    return obj

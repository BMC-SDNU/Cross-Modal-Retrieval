import cPickle as pickle


def save_pickle(outpath, obj):
    with open(outpath, 'wb') as fp:
        pickle.dump(obj, fp, protocol=2)


def load_pickle(outpath):
    with open(outpath, 'rb') as fp:
        return pickle.load(fp)


def load_json(path):
    import json
    from pprint import pprint

    with open(path) as data_file:    
        data = json.load(data_file)
    return data


def save_json(path, obj):
    import json
    with open(path, 'w') as outfile:
        json.dump(obj, outfile, indent=4)
 

def open_txt_file(path):
    file = open(path, 'r')
    res = []
    for line in file:
        res.append(line.replace("\n",""))
    return res    


def create_path(path):
    import os
    try:
        os.makedirs(path)
        return True
    except OSError, oe:        
        return False

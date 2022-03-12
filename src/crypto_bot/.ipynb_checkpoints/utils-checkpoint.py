import os
import pickle


def save_obj(filename, obj):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    outfile = open(filename, 'wb')
    pickle.dump(obj, outfile)
    outfile.close()


def load_obj(filename):
    infile = open(filename, 'rb')
    obj = pickle.load(infile)
    infile.close()
    return obj
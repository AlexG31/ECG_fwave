#encoding:utf8
import os
import sys
import time
import json
import pdb
import bisect
import joblib
import numpy as np
import numpy.random as random
import random as pyrandom
from feature_extractor.feature_extractor import ECGfeatures
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from train_model import get_configuration



curfilepath =  os.path.realpath(__file__)
current_folderpath = os.path.dirname(curfilepath)

def testing(fs = 500):
    '''Changgeng testing.'''
    from ECGLoader import ECGLoader
    ecg = ECGLoader(fs, current_folderpath)

    with open('./models/fwave.mdl', 'rb') as fin:
        model = joblib.load(fin)
    testing_data = list()
    testing_infolist = list()
    # Add normal training samples
    for ind in xrange(0, 20):
        rawsig, diag_text, mat_file_name = ecg.load(ind, data_info_file_name = 'normal.json')
        print 'processing file:', mat_file_name 
        feature_extractor = ECGfeatures(rawsig, get_configuration())

        # Get QRS locations
        from dpi.DPI_QRS_Detector import DPI_QRS_Detector as DPI
        dpi = DPI()
        print 'Testing QRS locations ...'
        results = dpi.QRS_Detection(rawsig, fs)
        skip_QRS_count = 3
        for qrs_index in xrange(skip_QRS_count, len(results) - skip_QRS_count):
            qrs_pos = results[qrs_index]
            next_qrs_pos = results[qrs_index + 1]
            pos = (qrs_pos + next_qrs_pos) / 2.0

            feature_vector = feature_extractor.frompos(pos)
            testing_data.append(feature_vector)
            testing_infolist.append((pos, ind))
            break

    print 'length of testing_data:', len(testing_data)
    print 'length of testing_data[0]:', len(testing_data[0])
    start_time = time.time()
    results = model.predict(testing_data)
    print 'time cost: %f secs.' % (time.time() - start_time)

    with open('./output.json', 'w') as fout:
        json.dump(zip(testing_infolist, results), fout, indent = 4)
        print 'outputs to ./output.json'
    
    

if __name__ == '__main__':
    testing()

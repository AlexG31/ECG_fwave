#encoding:utf8
import os
import sys
import matplotlib.pyplot as plt
import glob
import json
from ECGLoader import ECGLoader

curfilepath =  os.path.realpath(__file__)
current_folderpath = os.path.dirname(curfilepath)

def get_configuration():
    '''Get configuration diction.'''
    current_file_path = os.path.realpath(__file__)
    current_folder = os.path.dirname(current_file_path)
    conf = dict(
            fs = 250,
            winlen_ratio_to_fs = 3,
            WT_LEVEL = 6
            )

    conf['random_pattern_path'] = os.path.join(current_folderpath, "models/random_pattern.json")
    return conf


def train_test(fs = 500):
    '''Train a fwave model.'''
    import codecs
    import bisect
    from feature_extractor.feature_extractor import ECGfeatures
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    ecg = ECGLoader(fs, current_folderpath)
    
    # Max length of R-R interval
    max_rr = 0
    training_data = list()
    labeled_records = glob.glob(os.path.join(current_folderpath, 'labeled-data', '*.json'))

    # Add fwave training samples
    for labeled_record_name in labeled_records:

        print 'processing file:', labeled_record_name
        with codecs.open(labeled_record_name, 'r', 'utf8') as fin:
            labeled_data = json.load(fin)
        diagnosis_text = labeled_data['diag_text']
        rawsig = ecg.loadMatwithName(labeled_data['mat_file_name'])

        configuration_info = get_configuration()
        feature_extractor = ECGfeatures(rawsig, configuration_info)

        rr_list = sorted(zip(*labeled_data['expertLabels'])[0])
        for pos in labeled_data['poslist']:
            feature_vector = feature_extractor.frompos(pos)
            training_data.append((feature_vector, 'fwave'))
            
            # Current R-R
            left_r_index = bisect.bisect_left(rr_list, pos)
            right_r_index = left_r_index + 1
            if right_r_index >= len(rr_list):
                print 'Warning: labeled fwave pos not in-between R-R!'
            else:
                max_rr = max(max_rr, abs(rr_list[left_r_index] - rr_list[right_r_index]))
            
            
            
            

    # Add normal training samples
    for ind in xrange(0, 20):
        rawsig, diag_text, mat_file_name = ecg.load(ind, data_info_file_name = 'normal.json')
        print 'processing file:', mat_file_name 
        # Get QRS locations
        from dpi.DPI_QRS_Detector import DPI_QRS_Detector as DPI
        dpi = DPI()
        print 'Testing QRS locations ...'
        results = dpi.QRS_Detection(rawsig, fs)
        skip_QRS_count = 3
        for qrs_index in xrange(skip_QRS_count, len(results) - skip_QRS_count):
            qrs_pos = results[qrs_index]
            next_qrs_pos = results[qrs_index + 1]
            normal_pos = (qrs_pos + next_qrs_pos) / 2.0

            # Get max R-R length
            max_rr = max(max_rr, abs(next_qrs_pos - qrs_pos))

            feature_vector = feature_extractor.frompos(normal_pos)
            training_data.append((feature_vector, 'normal'))
    
    model = RandomForestClassifier(
            30,
            max_depth = 30
            )
    X, y = zip(*training_data)

    print 'Fitting model ...'
    model.fit(X, y)

    print 'Maximum R-R interval is %d' % max_rr
    return model



def longest_RR():
    '''Get longest RR interval.'''
    train_test()
    
if __name__ == '__main__':
    longest_RR()
    # import joblib 
    # with open('./models/fwave.mdl', 'wb') as fout:
        # joblib.dump(train_test(), fout)
        # print 'File written to %s' % './models/fwave.mdl'
    

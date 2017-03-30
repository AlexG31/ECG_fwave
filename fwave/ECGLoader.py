#encoding:utf8
import os
import numpy as np
import sys

# ECG data from changgeng
class ECGLoader(object):
    def __init__(self, fs, current_folderpath):
        '''Loader for changgeng data.'''
        self.fs = fs
        self.current_folderpath = current_folderpath

    def load(self, record_index, data_info_file_name = 'fwave.json'):
        '''Return loaded signal info.'''
        return self.getSignal(data_info_file_name, record_index, 'II')

    def loadMatwithName(self, mat_file_name):
        '''Return loaded signal info.'''
        return self.getSignalwithName(mat_file_name, 'II')

    def getSize(self, jsonfilename):
        '''Get fangchan record count.'''
        import json
        import codecs

        matinfojson_filename = os.path.join(self.current_folderpath, 'changgeng', '%s' % jsonfilename)
        with codecs.open(matinfojson_filename, 'r', 'utf8') as fin:
            data = json.load(fin)

        # mat_rhythm is the data
        dlist = data['data']
        return len(dlist)

    def getSignal(self, jsonfilename, index, leadname):
        '''Get fangchan signal.'''
        import json
        import codecs
        import subprocess
        import scipy.io as sio

        matinfojson_filename = os.path.join(self.current_folderpath, 'changgeng', '%s' % jsonfilename)
        with codecs.open(matinfojson_filename, 'r', 'utf8') as fin:
            data = json.load(fin)

        # mat_rhythm is the data
        dlist = data['data']
        matpath = dlist[index]['mat_rhythm']
        diagnosis_text = dlist[index]['diagnose']

        mat_file_name = os.path.split(matpath)[-1]
        save_mat_filepath = os.path.join(self.current_folderpath, 'changgeng', 'data', mat_file_name)
        if (os.path.exists(save_mat_filepath) == False):
            subprocess.call(['scp', 'xinhe:%s' % matpath, save_mat_filepath])
        matdata = sio.loadmat(save_mat_filepath)
        raw_sig = np.squeeze(matdata[leadname])

        return (raw_sig, diagnosis_text, mat_file_name)

    def getSignalwithName(self, mat_file_name, leadname):
        '''Get fangchan signal.'''
        import scipy.io as sio

        save_mat_filepath = os.path.join(self.current_folderpath, 'changgeng', 'data', mat_file_name)
        if (os.path.exists(save_mat_filepath) == False):
            raise Exception('mat file not downloaded to local dist yet!')

        matdata = sio.loadmat(save_mat_filepath)
        raw_sig = np.squeeze(matdata[leadname])

        return raw_sig

import requests
import ast
import sys
import numpy as np
import pandas as pd
import json

from setting import Setting

setting = Setting()

pd.set_option('display.max_columns', None)

class Data_load():

    def data_download(self, pagenum):

        self.params = {
            'serviceKey': '7j3h+fl6vaFtpHwY/Iw2h89NlivKtImk/6GVOk/VYjTRyIqSAus3z5TmZ6B/WE57ruBFD24U9yHjGIFKsAEDgQ==',
            'type': '', 'numOfRows': '{datanum}'.format(datanum=setting.datanum),
            'pageNo': '{pageno}'.format(pageno=pagenum + 1), 'startDt': '2019-01-01', 'endDt': '2021-12-31'}

        response = requests.get(setting.url, params=self.params)
        result = response.content
        # string = str(result, 'utf-8')
        dict_str = result.decode("UTF-8")
        my_data = ast.literal_eval(dict_str)

        if pagenum == 0:
            self.data_list = []

        if pagenum == setting.pageno:
            setting.datanum = 662

        for i in range(0, setting.datanum, 1):
            data = my_data[0]['iotRoadFileList'][i]
            self.data_list.append(data)

        if pagenum == setting.pageno:
            data_concatenate = pd.DataFrame(self.data_list)
            data_concatenate.to_csv('./data_large.txt', sep='\t', index=False)
            print(data_concatenate)

    def data_loading(self):
        self.data = np.loadtxt('./data_for_MLP_update4.txt', skiprows=1)

    def data_preprocessing(self):

        data_length = len(self.data)

        input = self.data[:, setting.input_index]
        output = self.data[:, setting.output_index]

        maxval = np.max(input, axis=0)
        minval = np.min(input, axis=0)

        input = (input - minval) / (maxval - minval)

        input, output = self.unison_shuffled_copies(input, output)

        train_val_input = input[0:int(np.round(setting.trainval_rate*data_length)), :]
        train_val_label = output[0:int(np.round(setting.trainval_rate*data_length)), :]

        test_input = input[int(np.round(setting.trainval_rate * data_length)):, :]
        test_label = output[int(np.round(setting.trainval_rate*data_length)):, :]

        val_input = train_val_input[0:int(np.round(setting.val_rate * data_length)), :]
        val_label = train_val_label[0:int(np.round(setting.val_rate * data_length)), :]

        train_input = train_val_input[int(np.round(setting.val_rate * data_length)):, :]
        train_label = train_val_label[int(np.round(setting.val_rate * data_length)):, :]

        return train_input, train_label, val_input, val_label, test_input, test_label

    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
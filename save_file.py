import sys
import numpy as np
import pandas as pd

from setting import Setting

setting = Setting()

class Save_file():

    def setting_save(self):

        f = open('./save_file/setting_save.txt', 'w')

        f.write('input param index : {input} \n'.format(input=setting.input_index))
        f.write('output param index : {output} \n'.format(output=setting.output_index))
        f.write('one_hot on/off 0: off, 1: on : {num} \n'.format(num=setting.one_hot_onoff))
        f.write('train_val_rate : {num} \n'.format(num=setting.trainval_rate))
        f.write('val_rate : {num} \n'.format(num=setting.val_rate))
        f.write('batch size : {num} \n'.format(num=setting.batchsize))
        f.write('epochs : {num} \n'.format(num=setting.epochs))
        f.write('actv_name : {name} \n'.format(name=setting.actv_name))
        f.write('loss_name : {name} \n'.format(name=setting.loss))

        f.close()

    def history_save(self, train, test):

        # with open('./save_file/Train_val history.csv', mode='w') as f:
        #     train.to_csv(f)
        #
        # with open('./save_file/Test history.csv', mode='w') as f:
        #     test.to_csv(f)

        train.to_csv('./save_file/Train_val history.csv', index=False)
        test.to_csv('./save_file/Test history.csv', index=False)
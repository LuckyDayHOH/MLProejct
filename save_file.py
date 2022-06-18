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
        f.write('train_val_rate : {num} \n'.format(num=setting.trainval_rate))
        f.write('val_rate : {num} \n'.format(num=setting.val_rate))
        f.write('batch size : {num} \n'.format(num=setting.batchsize))
        f.write('nlayer : {num} \n'.format(num=setting.nlayer))
        f.write('nnode : {num} \n'.format(num=setting.nnode))
        f.write('epochs : {num} \n'.format(num=setting.epochs))
        f.write('actv_name : {name} \n'.format(name=setting.actv_name))
        f.write('loss_name : {name} \n'.format(name=setting.loss))

        f.close()

    def history_save(self, train, test):

        train.to_csv('./save_file/Train_val history.csv', index=False)
        test.to_csv('./save_file/Test history.csv', index=False)
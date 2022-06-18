import sys

class Setting():

    def __init__(self):

        self.url = 'http://apis.data.go.kr/C100006/zerocity/getIotRoadList'
        self.pageno = 255
        self.datanum = 1000
        self.data_download_onoff = 0  # 0 : off, 1 : on

        # self.input_index = [0, 1, 2, 3, 4, 9, 10, 11, 13, 14, 15]
        self.input_index = [0, 1, 4, 9, 10, 11, 13]
        self.output_index = [5, 6, 12]

        self.trainval_rate = 0.8
        self.val_rate = 0.2
        self.batchsize = 1000
        self.epochs = 100
        self.nlayer = 8
        self.nnode = 24
        self.actv_name = 'relu'
        self.loss = 'categorical_crossentropy'

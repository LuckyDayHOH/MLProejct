import os
import sys
import numpy as np
import pandas as pd
from setting import Setting
from data_load import Data_load
from mlp_load import MLP_load
from save_file import Save_file

setting = Setting()
data_load = Data_load()
mlp_load = MLP_load()
save_file = Save_file()

if setting.data_download_onoff == 1:
    for i in range(0, setting.pageno + 1, 1):
        data_load.data_download(i)
        print('Data download complete')
        sys.exit(0)

else:
    pass

data_load.data_loading()
check_val = 1

while check_val == 1:
    train_input, train_label, val_input, val_label, test_input, test_label = data_load.data_preprocessing()

    train_label, val_label, test_label = mlp_load.to_onehot(train_label, val_label, test_label)

    if len(train_label[0][0]) == len(test_label[0][0]) and len(train_label[0][0]) == len(val_label[0][0]):
        check_val = 0

    else:
        pass

model = mlp_load.create_model(len(train_label[0][0]))

# train
history = model.fit(train_input,
                    {'avg_road': train_label[:, 0], 'trinspct_road': train_label[:, 1], 'surfc_gripfc': train_label[:, 2]},
                    epochs=setting.epochs,
                    validation_data=(val_input, {'avg_road': val_label[:, 0], 'trinspct_road': val_label[:, 1], 'surfc_gripfc': val_label[:, 2]}),
                    batch_size=setting.batchsize)

# test
test_history = model.evaluate(test_input, {'avg_road': test_label[:, 0], 'trinspct_road': test_label[:, 1], 'surfc_gripfc': test_label[:, 2]})

history_df = pd.DataFrame(history.history)

test_history_df = pd.DataFrame([test_history])
test_history_df.columns = history_df.columns.tolist()[0: 1 + 2 * len(setting.output_index)]

# save
save_file.history_save(history_df, test_history_df)

save_file.setting_save()

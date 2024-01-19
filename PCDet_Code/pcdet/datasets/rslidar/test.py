import pickle

path_src = "/data01/home/scy0587/run/taole/pcdetlatest/data/RS_Datasets_npy/datasets/robosense_infos_train.pkl"

train_info = pickle.load(open(path_src, 'rb'))
print("hello")
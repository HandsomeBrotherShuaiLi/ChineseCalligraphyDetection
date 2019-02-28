from keras.callbacks import EarlyStopping, TensorBoard

from dlocr.ctpn import CTPN
from dlocr.ctpn.data_loader import DataLoader
from dlocr.ctpn import get_session
from dlocr.custom import LRScheduler, SingleModelCK
import keras.backend as K
import os

from dlocr.ctpn import default_ctpn_config_path
import numpy as np
import shutil
#D:\python_projects\ChineseCalligraphyDetection\data\anno_vali
valid_path='D:\python_projects\ChineseCalligraphyDetection\data\\anno_vali'
def movefile(anno_path):
    all_annos=os.listdir(anno_path)
    length=len(all_annos)
    all_annos=np.array(all_annos)
    np.random.shuffle(all_annos)
    valid_annos=all_annos[:int(0.2*length)]
    for i in valid_annos:
        shutil.move(os.path.join(anno_path,str(i)),valid_path)
        print(str(i)+' has moved to new folder')
    print(length,len(os.listdir(anno_path)))
    print(len(os.listdir(valid_path)))



if __name__ == '__main__':
    # print(os.listdir('../dlocr/logs'))
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-ie", "--initial_epoch", help="初始迭代数", default=0, type=int)
    parser.add_argument("--epochs", help="迭代数", default=20, type=int)
    parser.add_argument("--gpus", help="gpu的数量", default=1, type=int)
    parser.add_argument("--images_dir", help="图像位置", default="D:\python_projects\ChineseCalligraphyDetection\data\\train_img")
    parser.add_argument("--anno_dir", help="标注文件位置", default="D:\python_projects\ChineseCalligraphyDetection\data\\annotation")
    parser.add_argument("--config_file_path", help="模型配置文件位置",
                        default=default_ctpn_config_path)
    parser.add_argument("--weights_file_path", help="模型初始权重文件位置",
                        default=None)
    parser.add_argument("--save_weights_file_path", help="保存模型训练权重文件位置",
                        default=r'model/cv_weights-ctpnlstm-{epoch:02d}.hdf5')

    args = parser.parse_args()
    #movefile(args.anno_dir)

    K.set_session(get_session(0.8))
    config = CTPN.load_config(args.config_file_path)

    weights_file_path = args.weights_file_path
    if weights_file_path is not None:
        config["weight_path"] = weights_file_path
    config['num_gpu'] = args.gpus

    ctpn = CTPN(**config)

    save_weigths_file_path = args.save_weights_file_path

    if save_weigths_file_path is None:
        try:
            if not os.path.exists("model"):
                os.makedirs("model")
            save_weigths_file_path = "model/weights-ctpnlstm-{epoch:02d}.hdf5"
        except OSError:
            print('Error: Creating directory. ' + "model")

    train_data_loader = DataLoader(args.anno_dir, args.images_dir)
    valid_data_loader = DataLoader(valid_path, args.images_dir)

    checkpoint = SingleModelCK(save_weigths_file_path, model=ctpn.parallel_model, save_weights_only=False)
    earlystop = EarlyStopping(patience=10)
    log = TensorBoard(log_dir='../dlocr/logs', histogram_freq= 0 , write_graph=True, write_images=False,batch_size=1,
                      update_freq='batch')
    lr_scheduler = LRScheduler(lambda epoch, lr: lr / 2, watch="loss", watch_his_len=2)
    # print(data_loader.steps_per_epoch,args.epochs)
    his=ctpn.parallel_model.fit_generator(
        generator=train_data_loader.load_data(),
        steps_per_epoch=train_data_loader.steps_per_epoch,
        validation_data=valid_data_loader.load_data(),
        validation_steps=valid_data_loader.steps_per_epoch,
        epochs=args.epochs,
        callbacks=[checkpoint,earlystop,lr_scheduler,log],
        initial_epoch=args.initial_epoch
    )
    print('history',his.history)

    # ctpn.train(
    #            train_data_loader=train_data_loader,
    #            valid_data_loader=valid_data_loader,
    #            epochs=args.epochs,
    #            callbacks=[checkpoint, earlystop, lr_scheduler]
    # )
# tensorboard --logdir=D:\python_projects\ChineseCalligraphyDetection\dlocr\logs
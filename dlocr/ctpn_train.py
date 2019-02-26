from keras.callbacks import EarlyStopping, TensorBoard

from dlocr.ctpn import CTPN
from dlocr.ctpn.data_loader import DataLoader
from dlocr.ctpn import get_session
from dlocr.custom import LRScheduler, SingleModelCK
import keras.backend as K
import os

from dlocr.ctpn import default_ctpn_config_path


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
                        default=r'model/weights-ctpnlstm-{epoch:02d}.hdf5')

    args = parser.parse_args()

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

    data_loader = DataLoader(args.anno_dir, args.images_dir)

    checkpoint = SingleModelCK(save_weigths_file_path, model=ctpn.model, save_weights_only=True)
    earlystop = EarlyStopping(patience=10)
    log = TensorBoard(log_dir='../dlocr/logs', histogram_freq= 1 , write_graph=True, write_images=False,batch_size=1)
    lr_scheduler = LRScheduler(lambda epoch, lr: lr / 2, watch="loss", watch_his_len=2)
    # print(data_loader.steps_per_epoch,args.epochs)

    ctpn.train(data_loader.load_data(),
               epochs=args.epochs,
               steps_per_epoch=data_loader.steps_per_epoch,
               callbacks=[checkpoint, earlystop, lr_scheduler],
               initial_epoch=args.initial_epoch)

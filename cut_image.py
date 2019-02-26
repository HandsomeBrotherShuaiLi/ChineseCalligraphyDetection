import cv2,numpy as np
import pandas as pd
import threadpool,os
base_dir='data/train_img'
base_cut_dir="data/train_img_cut"
def test():
    global concat_csv, fail_img_path, chinese_dict, txtstr
    txtstr = str()
    concat_csv = pd.read_csv('data/original_csv/concat_train.csv')
    print('start the game....')
    fail_img_path = []
    temp = open('data/chinese/chinese_all.txt', mode='r', encoding='utf-8').readlines()[0].split(',')
    chinese_dict = {c: i for i, c in enumerate(temp)}
    # img=cv2.imread(os.path.join(base_dir,'img_calligraphy_00001_bg.jpg'))
    # imgtemp=img[20:50,60:100,:]
    # cv2.imwrite('test.jpg',imgtemp)
    cut_single_thread('img_calligraphy_00009_bg.jpg')
def cut_single_thread(img_path):
    """

    :param img_path: 图片的路径
    :return:
    """
    global concat_csv,fail_img_path,chinese_dict,txtstr
    res=concat_csv[concat_csv.FileName==img_path]
    if len(res)==0:
        print(img_path," can't find this filename in concat csv file")
        fail_img_path.append(img_path)
        return
    # print(img_path,' find {} line'.format(len(res)))

    img=cv2.imread(os.path.join(base_dir,img_path))
    index_list=list(res.index)
    print(index_list)
    for count in range(len(index_list)):
        print('start '+img_path.strip('.jpg')+'_'+str(count)+' cutting')
        x = np.array([res.loc[index_list[count], 'x1'], res.loc[index_list[count], 'x2'], res.loc[index_list[count], 'x3'], res.loc[index_list[count], 'x4']])
        y = np.array([res.loc[index_list[count], 'y1'], res.loc[index_list[count], 'y2'], res.loc[index_list[count], 'y3'], res.loc[index_list[count], 'y4']])
        xmin_int = x.min()
        xmax_int = x.max()
        ymin_int = y.min()
        ymax_int = y.max()
        img_temp=img[ymin_int:ymax_int,xmin_int:xmax_int,:]
        cv2.imwrite(os.path.join(base_cut_dir,img_path.strip('.jpg')+'_'+str(count)+'.jpg'),img_temp)
        txtstr+=img_path.strip('.jpg')+'_'+str(count)+'.jpg'
        text=res.loc[index_list[count],'text']
        for i in text:
            txtstr+=' '+str(chinese_dict[i])
        txtstr+='/n'
        print(img_path.strip('.jpg')+'_'+str(count)+'.jpg   Done!')

def cut_multi_thread():
    global concat_csv, fail_img_path, chinese_dict, txtstr
    txtstr=str()
    concat_csv=pd.read_csv('data/original_csv/concat_train.csv')
    print('start the game....')
    fail_img_path=[]
    temp=open('data/chinese/chinese_all.txt',mode='r',encoding='utf-8').readlines()[0].split(',')
    chinese_dict={c:i for i,c in enumerate(temp)}
    # print(chinese_dict)
    threadcount = 1024
    pool = threadpool.ThreadPool(threadcount)
    request = threadpool.makeRequests(cut_single_thread,
                                      os.listdir('data/train_img'))
    [pool.putRequest(req) for req in request]
    pool.wait()

    print('fail img path:',fail_img_path)
    with open('data/train_img_labels/labels.txt',mode='w',encoding='utf-8') as f:
        f.write(txtstr)
    with open('fail_img.txt','w',encoding='utf-8') as f:
        f.write(','.join(fail_img_path))
    print('ALL DONE!!!!!')

if __name__=='__main__':
    # test()
    cut_multi_thread()











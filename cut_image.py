cut_img_dir="D:\py_projects\data\\train_img_cut_new"
csv_path='D:\py_projects\data\original_csv\concat_train.csv'
labels_path='D:\py_projects\data\\train_img_labels\labels.txt'
chinese_path='D:\py_projects\data\chinese\chinese_all.txt'
import cv2,numpy as np
import pandas as pd
import threadpool,os
import operator
base_dir='D:\py_projects\data\\train_img'
base_cut_dir="D:\py_projects\data\\train_img_cut_new_plus"
def get_labels(img_path):
    """

    :param img_path: 原来完整图片的路径
    :return:
    """
    global concat_csv,fail_img_path,chinese_dict,txtstr
    if img_path.endswith('.jpg'):
        res = concat_csv[concat_csv.FileName == img_path]
        if len(res)==0:
            print('没找到'+img_path)
            fail_img_path.append('finding 0 file '+img_path)
            return
        else:
            index_list = list(res.index)
            for count in range(len(index_list)):
                name=img_path.replace('.jpg', '_' + str(count) + '.jpg')
                text = res.loc[index_list[count], 'text']
                for i in text:
                    name += ' ' + str(chinese_dict[i])
                print(name)
                txtstr.append(name)
    else:
        print('不是合适的格式',img_path)
        fail_img_path.append('format error '+img_path)
        return

def labels_multi_thread():
    global concat_csv, fail_img_path, chinese_dict, txtstr
    txtstr = []
    concat_csv = pd.read_csv('D:\py_projects\data\original_csv\concat_train.csv')
    print('start the game....')
    fail_img_path = []
    temp = open('D:\py_projects\data\chinese\chinese_all.txt', mode='r', encoding='utf-8').readlines()[0].split(',')
    chinese_dict = {c: i for i, c in enumerate(temp)}
    threadcount = 2000
    pool = threadpool.ThreadPool(threadcount)
    request = threadpool.makeRequests(get_labels,
                                      os.listdir(base_dir))
    [pool.putRequest(req) for req in request]
    pool.wait()
    with open('labels.txt',mode='w',encoding='utf-8') as f:
        f.write('\n'.join(txtstr))
    with open('fail_img_labels.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(fail_img_path))
    print('done!')




def testsize():
    cuted_img_size=len(os.listdir(cut_img_dir))
    csvfile=pd.read_csv(csv_path)
    if cuted_img_size==len(csvfile):
        print('the same size')
    else:
        print('the different size')

def test():
    global concat_csv, fail_img_path, chinese_dict, txtstr
    txtstr = str()
    concat_csv = pd.read_csv('D:\py_projects\data\original_csv\concat_train.csv')
    print('start the game....')
    fail_img_path = []
    temp = open('D:\py_projects\data\chinese\chinese_all.txt', mode='r', encoding='utf-8').readlines()[0].split(',')
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
    global concat_csv,fail_img_path,chinese_dict,txtstr,cuted_img_list,all_cut_imgname
    if img_path.endswith('.jpg'):
        res = concat_csv[concat_csv.FileName == img_path]
        if len(res) == 0:
            print(img_path, " can't find this filename in concat csv file")
            fail_img_path.append(img_path)
            return
        # print(img_path,' find {} line'.format(len(res)))

        try:
            img = cv2.imread(os.path.join(base_dir, img_path))
        except Exception as e:
            fail_img_path.append(str(e)+'_'+str(img_path))
            return
        index_list = list(res.index)
        # print(index_list)
        for count in range(len(index_list)):
            all_cut_imgname.append(img_path.replace('.jpg', '_' + str(count) + '.jpg'))
            # print('start ' + img_path.strip('.jpg') + '_' + str(count) + ' cutting')
            if img_path.replace('.jpg', '_' + str(count) + '.jpg') in cuted_img_list:
                print(img_path.replace('.jpg', '_' + str(count) + '.jpg'))
                pass
            else:
                print(img_path.replace('.jpg', '_' + str(count) + '.jpg')+' 不存在')
                try:
                    x = np.array(
                        [res.loc[index_list[count], 'x1'], res.loc[index_list[count], 'x2'],
                         res.loc[index_list[count], 'x3'],
                         res.loc[index_list[count], 'x4']])
                    y = np.array(
                        [res.loc[index_list[count], 'y1'], res.loc[index_list[count], 'y2'],
                         res.loc[index_list[count], 'y3'],
                         res.loc[index_list[count], 'y4']])
                    xmin_int = x.min()
                    xmax_int = x.max()
                    ymin_int = y.min()
                    ymax_int = y.max()
                    print(xmin_int,ymin_int,xmax_int,ymax_int)
                    img_temp = img[ymin_int:ymax_int, xmin_int:xmax_int, :]
                    cv2.imwrite(os.path.join(base_cut_dir, img_path.replace('.jpg', '_' + str(count) + '.jpg')),
                                img_temp)
                    print(img_path.replace('.jpg', '_' + str(count) + '.jpg')+'***写入完毕***')
                except Exception as e:
                    print(e)
                    fail_img_path.append(str(e) + '_' + str(img_path) + '_' + str(count))
                try:
                    temp_str= img_path.replace('.jpg', '_' + str(count) + '.jpg')
                    text = res.loc[index_list[count], 'text']
                    for i in text:
                        temp_str += ' ' + str(chinese_dict[i])
                    txtstr.append(temp_str)
                    print(txtstr)
                    print(img_path.replace('.jpg', '_' + str(count) + '.jpg') + '   Done!')
                except Exception as e:
                    print('text error')
                    fail_img_path.append('text error:' + img_path)
    else:
        fail_img_path.append(img_path)
        return


def cut_multi_thread():
    global concat_csv, fail_img_path, chinese_dict, txtstr,cuted_img_list,all_cut_imgname
    txtstr=[]
    all_cut_imgname=[]
    cuted_img_list=os.listdir(cut_img_dir)
    concat_csv=pd.read_csv('D:\py_projects\data\original_csv\concat_train.csv')
    print('start the game....')
    fail_img_path=[]
    temp=open('D:\py_projects\data\chinese\chinese_all.txt',mode='r',encoding='utf-8').readlines()[0].split(',')
    chinese_dict={c:i for i,c in enumerate(temp)}
    # print(chinese_dict)
    threadcount = 2000
    pool = threadpool.ThreadPool(threadcount)
    request = threadpool.makeRequests(cut_single_thread,
                                      os.listdir('D:\py_projects\data\\train_img'))
    [pool.putRequest(req) for req in request]
    pool.wait()

    print('fail img path:',fail_img_path)
    with open('D:\py_projects\data\\train_img_labels\labels.txt',mode='w',encoding='utf-8') as f:
        f.write('\n'.join(txtstr))
    with open('fail_img.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(fail_img_path))
    with open('all_cut_img_name.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(all_cut_imgname))

    print('ALL DONE!!!!!')




if __name__=='__main__':
    # a=[(1,2),('set',1),3]
    # print(a)

    # # test()
    labels_multi_thread()
    # cut_multi_thread()

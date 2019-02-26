import os
from xml.dom.minidom import Document
import cv2,pandas as pd,numpy as np
import threadpool
import xml.etree.ElementTree as ET
def txt2xml(img_folder_path,converted_xml_path,csv_path):
    """

    :param img_folder_path: 图片文件夹路径
    :param converted_xml_path: 转化成的xml文件路径
    :param csv_path: csv文件的路径
    :param dict_path: 生成的所有出现的字的字典
    :return:
    """
    chinese=set()
    for img_path in os.listdir(img_folder_path):
        if (img_path.endswith('.png') or img_path.endswith('.jpg')) and img_path!='1.png':
            doc = Document()
            annotation = doc.createElement('annotation')
            doc.appendChild(annotation)
            folder = doc.createElement('folder')
            annotation.appendChild(folder)
            folder_content = doc.createTextNode(img_folder_path)
            folder.appendChild(folder_content)

            filename = doc.createElement('filename')
            annotation.appendChild(filename)
            filename_content = doc.createTextNode(img_path)
            filename.appendChild(filename_content)

            """
            read img file 
            """
            img=cv2.imread(os.path.join(img_folder_path,img_path))
            h,w,c=img.shape
            size=doc.createElement('size')
            annotation.appendChild(size)
            width=doc.createElement('width')
            size.appendChild(width)
            width_content=doc.createTextNode(str(w))
            width.appendChild(width_content)

            height=doc.createElement('height')
            size.appendChild(height)
            height_content=doc.createTextNode(str(h))
            height.appendChild(height_content)

            channel=doc.createElement('depth')
            size.appendChild(channel)
            channel_txt=doc.createTextNode(str(c))
            channel.appendChild(channel_txt)

            df=pd.read_csv(csv_path)
            res=df[df.FileName==img_path]
            # print(res)
            for i in res.index:
                # print('i=',i)
                object_new=doc.createElement('object')
                annotation.appendChild(object_new)
                name=doc.createElement('name')
                object_new.appendChild(name)
                name_txt=doc.createTextNode(res.loc[i,'text'])
                name.appendChild(name_txt)

                for c in res.loc[i,'text']:
                    chinese.add(c)
                bndbox = doc.createElement('bndbox')
                object_new.appendChild(bndbox)
                """
                因为rpn是基于xmin xmax等四个坐标来做的，所以不得不把坐标位置修改成四元组
                所以这只能基于长方形训练，对于普通四边形不可行
                """
                x=np.array([res.loc[i,'x1'],res.loc[i,'x2'],res.loc[i,'x3'],res.loc[i,'x4']])
                y=np.array([res.loc[i,'y1'],res.loc[i,'y2'],res.loc[i,'y3'],res.loc[i,'y4']])

                xmin_int=x.min()
                xmax_int=x.max()
                ymin_int=y.min()
                ymax_int=y.max()

                xmin=doc.createElement('xmin')
                bndbox.appendChild(xmin)
                xmin_text=doc.createTextNode(str(xmin_int))
                xmin.appendChild(xmin_text)

                ymin = doc.createElement('ymin')
                bndbox.appendChild(ymin)
                ymin_text = doc.createTextNode(str(ymin_int))
                ymin.appendChild(ymin_text)

                xmax= doc.createElement('xmax')
                bndbox.appendChild(xmax)
                xmax_text = doc.createTextNode(str(xmax_int))
                xmax.appendChild(xmax_text)

                ymax = doc.createElement('ymax')
                bndbox.appendChild(ymax)
                ymax_text = doc.createTextNode(str(ymax_int))
                ymax.appendChild(ymax_text)

            xml_path=os.path.join(converted_xml_path,img_path.strip('.jpg')+'.xml')
            # temp=doc.toprettyxml(indent='\t',encoding='utf-8')
            # print(temp)
            # print('++++++++++++')
            with open(xml_path,'w',encoding='utf-8') as f:
                doc.writexml(f,indent='\t',encoding='utf-8',newl='\n',addindent='\t')
                print(xml_path+' done!')
    with open('data/chinese/chinese.txt','w') as f:
        f.write(','.join(list(chinese)))
    print('all done!')

def txt2xml_single_thread(img_path):
    img_folder_path='D:\python_projects\ChineseCalligraphyDetection\data\\train_img'
    csv_path='D:\python_projects\ChineseCalligraphyDetection\data\original_csv\concat_train.csv'
    converted_xml_path='D:\python_projects\ChineseCalligraphyDetection\data\\annotation'
    # global chinese
    if (img_path.endswith('.png') or img_path.endswith('.jpg')) and img_path != '1.png':
        doc = Document()
        annotation = doc.createElement('annotation')
        doc.appendChild(annotation)
        folder = doc.createElement('folder')
        annotation.appendChild(folder)
        folder_content = doc.createTextNode(img_folder_path)
        folder.appendChild(folder_content)

        filename = doc.createElement('filename')
        annotation.appendChild(filename)
        filename_content = doc.createTextNode(img_path)
        filename.appendChild(filename_content)

        """
        read img file 
        """
        img = cv2.imread(os.path.join(img_folder_path, img_path))
        h, w, c = img.shape
        size = doc.createElement('size')
        annotation.appendChild(size)
        width = doc.createElement('width')
        size.appendChild(width)
        width_content = doc.createTextNode(str(w))
        width.appendChild(width_content)

        height = doc.createElement('height')
        size.appendChild(height)
        height_content = doc.createTextNode(str(h))
        height.appendChild(height_content)

        channel = doc.createElement('depth')
        size.appendChild(channel)
        channel_txt = doc.createTextNode(str(c))
        channel.appendChild(channel_txt)

        df = pd.read_csv(csv_path)
        res = df[df.FileName == img_path]
        # print(res)
        for i in res.index:
            # print('i=',i)
            object_new = doc.createElement('object')
            annotation.appendChild(object_new)
            # name = doc.createElement('name')
            # object_new.appendChild(name)
            # name_txt = doc.createTextNode(res.loc[i, 'text'])
            # name.appendChild(name_txt)

            # for c in res.loc[i, 'text']:
            #     chinese.add(c)
            bndbox = doc.createElement('bndbox')
            object_new.appendChild(bndbox)
            """
            因为rpn是基于xmin xmax等四个坐标来做的，所以不得不把坐标位置修改成四元组
            所以这只能基于长方形训练，对于普通四边形不可行
            """
            x = np.array([res.loc[i, 'x1'], res.loc[i, 'x2'], res.loc[i, 'x3'], res.loc[i, 'x4']])
            y = np.array([res.loc[i, 'y1'], res.loc[i, 'y2'], res.loc[i, 'y3'], res.loc[i, 'y4']])

            xmin_int = x.min()
            xmax_int = x.max()
            ymin_int = y.min()
            ymax_int = y.max()

            xmin = doc.createElement('xmin')
            bndbox.appendChild(xmin)
            xmin_text = doc.createTextNode(str(xmin_int))
            xmin.appendChild(xmin_text)

            ymin = doc.createElement('ymin')
            bndbox.appendChild(ymin)
            ymin_text = doc.createTextNode(str(ymin_int))
            ymin.appendChild(ymin_text)

            xmax = doc.createElement('xmax')
            bndbox.appendChild(xmax)
            xmax_text = doc.createTextNode(str(xmax_int))
            xmax.appendChild(xmax_text)

            ymax = doc.createElement('ymax')
            bndbox.appendChild(ymax)
            ymax_text = doc.createTextNode(str(ymax_int))
            ymax.appendChild(ymax_text)

        xml_path = os.path.join(converted_xml_path, img_path.strip('.jpg') + 'g.xml')
        with open(xml_path, 'w', encoding='utf-8') as f:
            doc.writexml(f, indent='\t', encoding='utf-8', newl='\n', addindent='\t')
            print(xml_path + ' done!')

def csv2xml_multiThread():
    global chinese
    chinese=set()
    threadcount=1280
    pool = threadpool.ThreadPool(threadcount)
    print(os.listdir('D:\python_projects\ChineseCalligraphyDetection\data\\train_img'))
    request = threadpool.makeRequests(txt2xml_single_thread,
                                      os.listdir('D:\python_projects\ChineseCalligraphyDetection\data\\train_img'))
    [pool.putRequest(req) for req in request]
    pool.wait()
    with open('data/chinese/chinese.txt','w',encoding='utf-8') as f:
        f.write(','.join(list(chinese)))
    print('all done!')


def merge_chinese():
    f1 = open('data/chinese/chinese.txt', 'r', encoding='utf-8').readlines()
    f2 = open('data/chinese/chinese_2.txt', 'r', encoding='utf-8').readlines()
    train_character = set(f1[0].split(','))
    print(len(train_character))
    vailidate_character = set(f2[0].split(','))
    print(len(vailidate_character))
    for i in vailidate_character:
        train_character.add(i)
    print(len(train_character))
    with open('data/chinese/chinese_all.txt', 'w',encoding='utf-8') as f:
        f.write(','.join(list(train_character)))
def merge_csv():
    train_csv=pd.read_csv('D:\python_projects\huawei\\traindataset\\train_label.csv')
    validation_csv=pd.read_csv('D:\python_projects\huawei\\traindataset\\verify_label.csv')
    ans=pd.concat([train_csv,validation_csv],ignore_index=True)
    ans.to_csv('data/original_csv/concat_train.csv',index=False)
def handle_xml_bugs_single(xmlpath):
    base_dir='D:\python_projects\ChineseCalligraphyDetection\data\\annotation'
    tree = ET.parse(os.path.join(base_dir,xmlpath))
    root = tree.getroot()
    res=root.findall('object')
    if len(res)==0:
        print(xmlpath,'没有gbboxes，进行处理')
        imgpath=xmlpath.strip('.xml')+'.jpg'
        print('开始处理图片 ',imgpath)
        txt2xml_single_thread(imgpath)
def handle_xml_bugs():
    threadcount = 1280
    pool = threadpool.ThreadPool(threadcount)
    request = threadpool.makeRequests(handle_xml_bugs_single,
                                      os.listdir('D:\python_projects\ChineseCalligraphyDetection\data\\annotation'))
    [pool.putRequest(req) for req in request]
    pool.wait()
    print('all done!')

if __name__=='__main__':
    handle_xml_bugs()





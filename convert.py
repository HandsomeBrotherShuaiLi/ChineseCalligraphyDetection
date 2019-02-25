import os
from xml.dom.minidom import Document
import cv2,pandas as pd,numpy as np
import threadpool
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
    img_folder_path='D:\python_projects\huawei\\traindataset\\trian_Image'
    csv_path='D:\python_projects\huawei\\traindataset\\train_label.csv'
    converted_xml_path='D:\python_projects\huawei\\traindataset\\train_anno'
    global chinese
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
            name = doc.createElement('name')
            object_new.appendChild(name)
            name_txt = doc.createTextNode(res.loc[i, 'text'])
            name.appendChild(name_txt)

            for c in res.loc[i, 'text']:
                chinese.add(c)
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

        xml_path = os.path.join(converted_xml_path, img_path.strip('.jpg') + '.xml')
        # temp=doc.toprettyxml(indent='\t',encoding='utf-8')
        # print(temp)
        # print('++++++++++++')
        with open(xml_path, 'w', encoding='utf-8') as f:
            doc.writexml(f, indent='\t', encoding='utf-8', newl='\n', addindent='\t')
            print(xml_path + ' done!')

def csv2xml_multiThread():
    global chinese
    chinese=set()
    threadcount=128
    pool = threadpool.ThreadPool(threadcount)
    request = threadpool.makeRequests(txt2xml_single_thread,
                                      os.listdir('D:\python_projects\huawei\\traindataset\\trian_Image'))
    [pool.putRequest(req) for req in request]
    pool.wait()
    with open('data/chinese/chinese.txt','w',encoding='utf-8') as f:
        f.write(','.join(list(chinese)))
    print('all done!')


if __name__=='__main__':
    csv2xml_multiThread()
    # txt2xml('D:\python_projects\huawei\\traindataset\\trian_Image',
    #         'D:\python_projects\huawei\\traindataset\\train_anno',
    #         'D:\python_projects\huawei\\traindataset\\train_label.csv')




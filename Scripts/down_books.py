import requests
import os
import time

def down_book(uri,pos,PAGENUM,root):
    """
    :param uri: the base uri 
    :param pos: the posix
    :param PAGENUM: total page number
    :param root: the saved root directory name
    :return: none
    """
    for id in range(1,PAGENUM):
        file_path = '%s/%3d%s'%(root,id,pos)
        try:
            if not os.path.exists(root):
                os.mkdir(root)
            if not os.path.exists(file_path):
                url = '%s%03d%s'%(uri,id,pos)
                print(url)
                r = requests.get(url)
                r.raise_for_status()
                with open(file_path,'wb') as f:
                    f.write(r.content)
                print('done..')
            else:
                print('exists')
        except Exception as e:
            print('err: '+str(e))

if __name__ == '__main__':
    # http://d.yuwenziyuan.com/rjb/UploadFile/dzkb/8s_2017/050.jpg
    uri = 'http://d.yuwenziyuan.com/rjb/UploadFile/dzkb/8x_new/'
    pos = '.jpg'
    PAGENUM = 142
    root = '/Users/Zoking/Downloads/语文8下'
    down_book(uri, pos, PAGENUM, root)

from PIL import Image
import os

def merge_book(root,pos,PAGENUM):
    """
    :param root: root directory name
    :param pos: file posix
    :param PAGENUM: total pageNum
    :return: none
    """
    img_list = []
    im = Image.open('%s/%3d%s'%(root,1,pos))
    for id in range(2,PAGENUM):
        file_path = '%s/%3d%s'%(root,id,pos)
        try:
            if not os.path.exists(root):
                os.mkdir(root)
            if os.path.exists(file_path):
                img = Image.open(file_path)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img_list.append(img)
            else:
                print('file not exists')
        except Exception as e:
            print('err: '+str(e))
    file = '%s/%s'%(root,'语文8下.pdf')
    if os.path.exists(file):
        print('exists...')
    else:
        im.save(file,'PDF',resolution=100.0,save_all=True,append_images=img_list)
    print('done...')

if __name__ == '__main__':
    root = '/Users/Zoking/Downloads/语文8下'
    pos = '.jpg'
    PAGENUM = 142
    merge_book(root, pos, PAGENUM)

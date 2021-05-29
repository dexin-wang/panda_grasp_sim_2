import math
import cv2
import os
import zipfile
import numpy as np


def quaternion_to_rotation_matrix(q):  # x, y ,z ,w
    rot_matrix = np.array(
        [[1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2])],
         [2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] - q[3] * q[0])],
         [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 * (q[0] * q[0] + q[1] * q[1])]],
        dtype=np.float)
    return rot_matrix


def getTransfMat(offset, rotate):
    """
    将平移向量和旋转矩阵合并为变换矩阵
    offset: (x, y, z)
    rotate: 旋转矩阵
    """
    mat = np.array([
        [rotate[0, 0], rotate[0, 1], rotate[0, 2], offset[0]], 
        [rotate[1, 0], rotate[1, 1], rotate[1, 2], offset[1]], 
        [rotate[2, 0], rotate[2, 1], rotate[2, 2], offset[2]],
        [0, 0, 0, 1.] 
    ])
    return mat

def depth2Gray(im_depth):
    """
    将深度图转至8位灰度图
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max
    return (im_depth * k + b).astype(np.uint8)

# def resize(img):
#     returncv2.resize(img, (1000, 1000))

def distancePt(pt1, pt2):
    """
    计算两点之间的欧氏距离
    pt: [row, col] 或 [x, y]
    return: float
    """
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

def distancePt3d(pt1, pt2):
    """
    计算两点之间的欧氏距离
    pt: [x, y, z]
    return: float
    """
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2) ** 0.5


def calcAngleOfPts(pt1, pt2):
    """
    计算从pt1到pt2的逆时针夹角 [0, 2pi)
    
    pt: [x, y] 二维坐标系中的坐标，不是图像坐标系的坐标
    
    return: 弧度
    """
    dy = pt2[1] - pt1[1]
    dx = pt2[0] - pt1[0]
    return (math.atan2(dy, dx) + 2*math.pi) % (2*math.pi)
    

def radians_TO_angle(radians):
    """
    弧度转角度
    """
    return 180 * radians / math.pi

def angle_TO_radians(angle):
    """
    角度转弧度
    """
    return math.pi * angle / 180

def depth3C(depth):
    """
    将深度图转化为3通道 np.uint8类型
    """
    depth_3c = depth[..., np.newaxis]
    depth_3c = np.concatenate((depth_3c, depth_3c, depth_3c), axis=2)
    return depth_3c.astype(np.uint8)

def zip_file(filedir):
    """
    压缩文件夹至同名zip文件
    """
    file_news = filedir + '.zip'
    # 如果已经有压缩的文件，删除
    if os.path.exists(file_news):
        os.remove(file_news)

    z = zipfile.ZipFile(file_news,'w',zipfile.ZIP_DEFLATED) #参数一：文件夹名
    for dirpath, dirnames, filenames in os.walk(filedir):
        fpath = dirpath.replace(filedir,'') #这一句很重要，不replace的话，就从根目录开始复制
        fpath = fpath and fpath + os.sep or ''#这句话理解我也点郁闷，实现当前文件夹以及包含的所有文件的压缩
        for filename in filenames:
            z.write(os.path.join(dirpath, filename),fpath+filename)
    z.close()


def unzip(file_name):
    """
    解压缩zip文件至同名文件夹
    """
    zip_ref = zipfile.ZipFile(file_name) # 创建zip 对象
    os.mkdir(file_name.replace(".zip","")) # 创建同名子文件夹
    zip_ref.extractall(file_name.replace(".zip","")) # 解压zip文件内容到子文件夹
    zip_ref.close() # 关闭zip文件


if __name__ == "__main__":
    filename = "D:/research/grasp_detection/Grasp_Correction/code-1/img/img_urdf_1"  #要压缩的文件夹路径
    # zip_file(filename)          # 压缩
    unzip(filename + '.zip')    # 解压
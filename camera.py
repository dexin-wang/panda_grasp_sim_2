import cv2
import math
import time
import numpy as np
import scipy.io as scio

# 图像参数
HEIGHT = 480
WIDTH = 640


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

def eulerAnglesToRotationMatrix(theta):
    """
    欧拉角转旋转矩阵
    theta: [r, p, y]
    """
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

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


class Camera:
    def __init__(self):
        """
        初始化相机参数，计算相机内参
        """
        self.fov = 60   # 垂直视场
        self.length = 0.7
        self.H = self.length * math.tan(angle_TO_radians(self.fov/2))   # 图像上方点的到中心点的实际距离 m
        self.W = WIDTH * self.H / HEIGHT     # 图像右方点的到中心点的实际距离 m
        # 计算 f/dx 和 f/dy
        self.A = (HEIGHT / 2) * self.length / self.H
        # 计算内参
        self.InMatrix = np.array([[self.A, 0, WIDTH/2 - 0.5], [0, self.A, HEIGHT/2 - 0.5], [0, 0, 1]], dtype=np.float)
        # 计算世界坐标系->相机坐标系的转换矩阵 4*4
        # 欧拉角: (pi, 0, 0)    平移(0, 0, 0.7)
        rotMat = eulerAnglesToRotationMatrix([math.pi, 0, 0])
        self.transMat = getTransfMat([0, 0, 0.7], rotMat)

    
    def img2camera(self, pt, dep):
        """
        获取像素点pt在相机坐标系中的坐标
        pt: [x, y]
        dep: 深度值

        return: [x, y, z]
        """
        pt_in_img = np.array([[pt[0]], [pt[1]], [1]], dtype=np.float)
        ret = np.matmul(np.linalg.inv(self.InMatrix), pt_in_img) * dep
        return list(ret.reshape((3,)))
        # print('坐标 = ', ret)
    
    def camera2img(self, coord):
        """
        将相机坐标系中的点转换至图像
        coord: [x, y, z]

        return: [row, col]
        """
        z = coord[2]
        coord = np.array(coord).reshape((3, 1))
        rc = (np.matmul(self.InMatrix, coord) / z).reshape((3,))

        return list(rc)[:-1]

    def length_TO_pixels(self, l, dep):
        """
        与相机距离为dep的平面上 有条线，长l，获取这条线在图像中的像素长度
        l: m
        dep: m
        """
        return l * self.A / dep
    
    def camera2world(self, coord):
        """
        获取相机坐标系中的点在世界坐标系中的坐标
        corrd: [x, y, z]

        return: [x, y, z]
        """
        coord.append(1.)
        coord = np.array(coord).reshape((4, 1))
        coord_new = np.matmul(self.transMat, coord).reshape((4,))
        return list(coord_new)[:-1]
    
    def world2camera(self, coord):
        """
        获取世界坐标系中的点在相机坐标系中的坐标
        corrd: [x, y, z]

        return: [x, y, z]
        """
        coord.append(1.)
        coord = np.array(coord).reshape((4, 1))
        coord_new = np.matmul(np.linalg.inv(self.transMat), coord).reshape((4,))
        return list(coord_new)[:-1]

    def world2img(self, coord):
        """
        获取世界坐标系中的点在图像中的坐标
        corrd: [x, y, z]

        return: [row, col]
        """
        # 转到相机坐标系
        coord = self.world2camera(coord)
        # 转到图像
        pt = self.camera2img(coord) # [y, x]
        return [int(pt[1]), int(pt[0])]

    def img2world(self, pt, dep):
        """
        获取像素点的世界坐标
        pt: [x, y]
        dep: 深度值 m
        """
        coordInCamera = self.img2camera(pt, dep)
        return self.camera2world(coordInCamera)



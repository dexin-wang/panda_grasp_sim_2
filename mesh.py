import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import path


GRASP_MAX_W = 0.08     # 最大张开宽度


def clacPlane(pt1, pt2, pt3):
    """
    根据三点计算平面方程 ax+by+cz+d=0
    pts: [[x, y, z], [x, y, z], [x, y, z]]
    return: A B C   z=Ax+By+C
    """
    
    a = (pt2[1]-pt1[1])*(pt3[2]-pt1[2]) - (pt2[2]-pt1[2])*(pt3[1]-pt1[1])
    b = (pt2[2]-pt1[2])*(pt3[0]-pt1[0]) - (pt2[0]-pt1[0])*(pt3[2]-pt1[2])
    c = (pt2[0]-pt1[0])*(pt3[1]-pt1[1]) - (pt2[1]-pt1[1])*(pt3[0]-pt1[0])
    d = 0 - (a * pt1[0] + b * pt1[1] + c * pt1[2])

    return a, b, c, d

    # if a == 0 or b == 0 or c == 0 or d == 0:
    #     print('a = ', a)
    #     print('b = ', b)
    #     print('c = ', c)
    #     print('d = ', d)
    # return np.array([-1*a/c, -1*b/c, -1*d/c])

def ptsInTriangle(pt1, pt2, pt3):
    """
    获取pt1 pt2 pt3 组成的三角形内的坐标点
    pt1: float [x, y]
    """
    p = path.Path([pt1, pt2, pt3])

    min_x = int(min(pt1[0], pt2[0], pt3[0]))
    max_x = int(max(pt1[0], pt2[0], pt3[0]))
    min_y = int(min(pt1[1], pt2[1], pt3[1]))
    max_y = int(max(pt1[1], pt2[1], pt3[1]))

    pts = []
    for x in range(min_x, max_x+1):
        for y in range(min_y, max_y+1):
            if p.contains_points([(x, y)])[0]:
                pts.append([x, y])
    
    return pts


class Mesh:
    """
    mesh 类，读取obj文件，坐标转换，生成空间深度图
    """
    def __init__(self, filename, scale=-1):
        """
        读取obj文件，获取v 和 f
        只用于读取EGAD数据集的obj文件

        filename: obj文件名
        scale: 物体缩放尺度,
            int: 缩放scale倍
            -1 : 自动设置scale，使外接矩形框的中间边不超过抓取器宽度(0.07)的80% scale最大为0.001
        """
        assert scale == -1 or scale > 0
        # print(filename)
        if scale > 0:
            self._scale = scale
        else:
            self._scale = 1

        with open(filename) as file:
            self.points = []
            self.faces = []
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    self.points.append((float(strs[1])*self._scale, float(strs[2])*self._scale, float(strs[3])*self._scale))
                if strs[0] == "f":
                    if strs[1].count('//'):
                        idx1, idx2, idx3 = strs[1].index('//'), strs[2].index('//'), strs[3].index('//')
                        self.faces.append((int(strs[1][:idx1]), int(strs[2][:idx2]), int(strs[3][:idx3])))
                    else:
                        self.faces.append((int(strs[1]), int(strs[2]), int(strs[3])))
        
        self.points = np.array(self.points)
        self.faces = np.array(self.faces, dtype=np.int64)
        if scale == -1:
            self._scale = self.get_scale()
            self.points = self.points * self._scale


    def min_z(self):
        """
        返回最小的z坐标
        """
        return np.min(self.points[:, 2])


    def get_scale(self):
        """
        自适应设置scale
        """
        d_x = np.max(self.points[:, 0]) - np.min(self.points[:, 0])
        d_y = np.max(self.points[:, 1]) - np.min(self.points[:, 1])
        d_z = np.max(self.points[:, 2]) - np.min(self.points[:, 2])
        ds = [d_x, d_y, d_z]
        ds.sort()
        scale = (GRASP_MAX_W - 0.01) * 0.8 / ds[1]
        if scale > 0.001:
            scale = 0.001
        
        return scale
    

    def scale(self):
        return self._scale
    

    def calcCenterPt(self):
        """
        计算mesh的中心点坐标 
        return: [x, y, z]
        """
        return np.mean(self.points, axis=0)

    
    def transform(self, mat):
        """
        根据旋转矩阵调整顶点坐标
        """
        points = self.points.T  # 转置
        ones = np.ones((1, points.shape[1]))
        points = np.vstack((points, ones))
        # 转换
        # print(mat.shape)
        # print(points.shape)
        new_points = np.matmul(mat, points)[:-1, :]
        self.points = new_points.T  # 转置  (n, 3)


if __name__ == "__main__":
    p = path.Path([(0, 0), (0.001, 0), (0, 0.001)])
    ret = p.contains_points([(0.00001, 0.00001)])[0]
    print(ret)




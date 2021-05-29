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
    
    def renderTableImg(self, mask_id, size=(0.8, 0.8), unit=0.001):
        """
        渲染相对于水平面的深度图和 obj mask，每个点之间的间隔为0.5mm
        size: (h, w) 单位 m

        算法流程:   
            方法1: 计算每个三角网格所在的平面方程，计算位于三角区域内的点(x,y)，带入平面方程，得到深度z
            方法2: 根据mesh顶点计算空间中的离散深度，对于在物体区域内(前述离散点的外包络)但深度为0的点，根据离它最近的三个点的插值计算深度
        """

        # 初始化深度图像
        depth_map = np.zeros((int(size[0] / unit), int(size[1] / unit)), dtype=np.float)
        for face in self.faces:
            pt1 = self.points[face[0] - 1]  # xyz m
            pt2 = self.points[face[1] - 1]
            pt3 = self.points[face[2] - 1]
            # 计算三角网格所在平面方程 xyz -> 平面
            plane_a, plane_b, plane_c, plane_d = clacPlane(pt1, pt2, pt3)    # ABC  Ax+By+C=z
            if plane_c == 0:
                continue
            plane = np.array([-1*plane_a/plane_c, -1*plane_b/plane_c, -1*plane_d/plane_c])

            # 将三角坐标转化为像素坐标
            pt1_pixel = [pt1[0] / unit, pt1[1] / unit]  # xy 像素 float
            pt2_pixel = [pt2[0] / unit, pt2[1] / unit]
            pt3_pixel = [pt3[0] / unit, pt3[1] / unit]
            # 获取三角形内的像素坐标 (x,y)
            pts_pixel = ptsInTriangle(pt1_pixel, pt2_pixel, pt3_pixel)

            if len(pts_pixel) == 0:
                continue

            # 像素坐标转化为实际坐标 m
            pts = np.array(pts_pixel, dtype=np.float) * unit  # (n, 2) m
            # 带入平面方程，计算深度 m
            ones = np.ones((pts.shape[0], 1))
            pts = np.hstack((pts, ones))
            depth = np.matmul(pts, plane.reshape((3,1))).reshape((-1,))
            # 将像素坐标pts_pixel转换到图像坐标系下
            pts_pixel = np.array(pts_pixel, dtype=np.int64)
            xs, ys = pts_pixel[:, 0], pts_pixel[:, 1]
            rows = ys * -1 + int(round(depth_map.shape[0] / 2))
            cols = xs + int(round(depth_map.shape[0] / 2))
            # 生成深度图像 pts_pixel depth
            depth_map[rows, cols] = np.maximum(depth_map[rows, cols], depth)
        
        return depth_map


# 绘制3D散点
# fig = plt.figure()
# ax1 = plt.axes(projection='3d')

# xs = points[:, 0]
# ys = points[:, 1]
# zs = points[:, 2]

# ax1.scatter3D(xs,ys,zs, cmap='Blues')  #绘制散点图
# plt.show()

# print(points)


if __name__ == "__main__":
    p = path.Path([(0, 0), (0.001, 0), (0, 0.001)])
    ret = p.contains_points([(0.00001, 0.00001)])[0]
    print(ret)




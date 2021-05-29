# -*- coding: UTF-8 -*-
"""===============================================
@Author : wangdx
@Date   : 2020/9/1 21:37
==============================================="""
import numpy as np
import cv2
import math
import scipy.io as scio


HEIGHT = 480
WIDTH = 640
GRASP_WIDTH_MAX = 0.08  # m
RADIO = 612.0



def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - round((angle + math.pi) // (2 * math.pi)) * 2 * math.pi

def drawGrasps(img, grasps, mode='line'):
    """
    绘制grasp
    file: img路径
    grasps: list()	元素是 [row, col, angle, width]
    angle: 弧度
    width: 单位 像素
    mode: 显示模式 'line' or 'region'
    """

    num = len(grasps)
    for i, grasp in enumerate(grasps):
        row, col, angle, width = grasp

        if mode == 'line':
            width = width / 2
            angle2 = calcAngle2(angle)
            k = math.tan(angle)
            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx
            if angle < math.pi:
                cv2.line(img, (col, row), (round(col + dx), round(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (round(col - dx), round(row + dy)), (0, 0, 255), 1)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (round(col + dx), round(row - dy)), (0, 0, 255), 1)
            else:
                cv2.line(img, (col, row), (round(col - dx), round(row + dy)), (0, 0, 255), 1)

        color_b = 255 / num * i
        color_r = 0
        color_g = -255 / num * i + 255

        if mode == 'line':
            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)
        else:
            img[row, col] = [color_b, color_g, color_r]
        

    return img


def imrotate(img,
             angle,
             center=None,
             scale=1.0,
             flag=cv2.INTER_NEAREST,
             border_value=0,
             auto_bound=False):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.

    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=flag, borderValue=border_value)
    return rotated


def calcEndPoints(pt, angle, width):
    """
    计算抓取端点(不分次序)

    params: 
        pt: (x, y) / (col, row)
        angle: 弧度
        width: 宽度
    """
    x, y = pt[0], pt[1]*-1
    dx = math.cos(angle) * width / 2.
    dy = math.sin(angle) * width / 2.

    pt1_x = int(x + dx)
    pt1_y = int(y + dy) * -1
    pt2_x = int(x - dx)
    pt2_y = int(y - dy) * -1

    return (pt1_x, pt1_y), (pt2_x, pt2_y)


def isPtInSquare(pt, center, size):
    """判断pt是否在正方形内，正方形的中心点是center, 边长是size
    
    params:
        pt: (x, y)
        center: (x, y)
        size: int
    """
    crop_t = int(center[1] - size/2)
    crop_b = int(center[1] + size/2)
    crop_l = int(center[0] - size/2)
    crop_r = int(center[0] + size/2)

    if crop_l <= pt[0] <= crop_r and crop_t <= pt[1] <= crop_b:
        return True
    return False
    


class GraspMat:
    def __init__(self, path):
        """
        file: *grasp.txt文件
        """
        # 读取抓取标签mat
        self.grasp_point_map = scio.loadmat(path + '/grasp_18/grasp_point_map.mat')['A']     # (h, w)
        self.grasp_angle_map = scio.loadmat(path + '/grasp_18/grasp_angle_map.mat')['A'].transpose((2, 0, 1))     # (h, w, bins)
        self.grasp_width_map = scio.loadmat(path + '/grasp_18/grasp_width_map.mat')['A'].transpose((2, 0, 1))     # (h, w, bins)
        self.grasp_depth_map = scio.loadmat(path + '/grasp_18/grasp_depth_map.mat')['A'].transpose((2, 0, 1))     # (h, w, bins)

        self.angle_k = self.grasp_angle_map.shape[0]
        self.height = self.grasp_angle_map.shape[1]
        self.width = self.grasp_angle_map.shape[2]
        

    def rotate(self, angle, center):
        """
        最近邻插值旋转, 每次旋转新建变量，不要修改原始标签变量

        angle: 逆时针旋转 角度 
        center: 旋转中心 (x, y)
        """
        assert angle % (180. / self.angle_k) == 0

        # imrotate 是顺时针旋转
        rota_r = -1 * angle
        self.grasp_point_map_rot = imrotate(self.grasp_point_map, rota_r, center)
        self.grasp_angle_map_rot = np.stack([imrotate(grasp_angle_map, rota_r, center) for grasp_angle_map in self.grasp_angle_map])
        self.grasp_width_map_rot = np.stack([imrotate(grasp_width_map, rota_r, center) for grasp_width_map in self.grasp_width_map])
        self.grasp_depth_map_rot = np.stack([imrotate(grasp_depth_map, rota_r, center) for grasp_depth_map in self.grasp_depth_map])

        # 逆时针旋转rota
        # 根据rota对grasp_angle_map进行平移
        # !!! 同时要对 grasp_width_map 和 grasp_depth_map 进行平移 
        offset = int(angle / (180. / self.angle_k))

        # offset 为正数时，列表下移；offset为负数时，列表上移
        self.grasp_angle_map_new = np.zeros_like(self.grasp_angle_map_rot)
        self.grasp_width_map_new = np.zeros_like(self.grasp_width_map_rot)
        self.grasp_depth_map_new = np.zeros_like(self.grasp_depth_map_rot)
        if offset != 0:  # 下移
            self.grasp_angle_map_new[:offset, :, :] = self.grasp_angle_map_rot[-1*offset:, :, :]    # 平移抓取角
            self.grasp_angle_map_new[offset:, :, :] = self.grasp_angle_map_rot[:-1*offset, :, :]

            self.grasp_width_map_new[:offset, :, :] = self.grasp_width_map_rot[-1*offset:, :, :]    # 抓取宽度
            self.grasp_width_map_new[offset:, :, :] = self.grasp_width_map_rot[:-1*offset, :, :]

            self.grasp_depth_map_new[:offset, :, :] = self.grasp_depth_map_rot[-1*offset:, :, :]    # 抓取深度
            self.grasp_depth_map_new[offset:, :, :] = self.grasp_depth_map_rot[:-1*offset, :, :]
        else:
            self.grasp_angle_map_new[:, :, :] = self.grasp_angle_map_rot[:, :, :]
            self.grasp_width_map_new[:, :, :] = self.grasp_width_map_rot[:, :, :]
            self.grasp_depth_map_new[:, :, :] = self.grasp_depth_map_rot[:, :, :]
        
        self.grasp_angle_map_rot[:, :, :] = self.grasp_angle_map_new[:, :, :]
        self.grasp_width_map_rot[:, :, :] = self.grasp_width_map_new[:, :, :]
        self.grasp_depth_map_rot[:, :, :] = self.grasp_depth_map_new[:, :, :]


    def getCorrectGrasp(self, pt, size):
        """
        搜索最接近pt的抓取标签, 满足抓取线全部在正方形内
        相似度: 抓取点偏差+抓取角(弧度)偏差 dx+dy+dr  偏差最小且抓取线全部位于图像块内

        param:
            pt: 待修正的抓取点 (row, col)
            size: 正方形边长 int
        return: 
            抓取修正(除了抓取宽度和深度)
            [drow, dcol, dangle_bin, width, depth]  注意:dangle_bin
        """
        pt_row, pt_col = pt
        # 检索抓取点在正方形内的抓取配置
        crop_t = int(pt_row - size/2)
        crop_b = int(pt_row + size/2)
        crop_l = int(pt_col - size/2)
        crop_r = int(pt_col + size/2)
        segmask = np.zeros((self.height, self.width), dtype=np.uint8)
        segmask[crop_t:crop_b, crop_l:crop_r] = np.ones((size, size), dtype=np.uint8)

        grasp_candidate_pts = np.where(self.grasp_point_map_rot > 0)
        if grasp_candidate_pts[0].shape[0] == 0:
            return None
        grasp_candidate_pts = np.c_[grasp_candidate_pts[0], grasp_candidate_pts[1]] # shape=(n,2) row, col
        grasp_candidate_pts = np.array( [p for p in grasp_candidate_pts if np.any(segmask[p[0], p[1]] > 0)] )
        if grasp_candidate_pts.shape[0] == 0:
            return None

        min_metric = 1000
        grasp_correction = None     # 存储抓取修正
        for grasp_pt in grasp_candidate_pts:
            grasp_row, grasp_col = grasp_pt
            
            grasp_angle_bins = np.where(self.grasp_angle_map_rot[:, grasp_row, grasp_col] > 0)[0]
            for grasp_angle_bin in grasp_angle_bins:
                grasp_angle = grasp_angle_bin / self.angle_k * math.pi  # 抓取角 弧度
                grasp_width = self.grasp_width_map_rot[grasp_angle_bin, grasp_row, grasp_col]   # 抓取宽度 米
                grasp_depth = self.grasp_depth_map_rot[grasp_angle_bin, grasp_row, grasp_col]   # 抓取深度 米
                # 1 根据抓取线筛选
                # 计算两个抓取端点坐标
                pt1, pt2 = calcEndPoints((grasp_col, grasp_row), grasp_angle, grasp_width * RADIO)
                # 判断两个抓取端点是否在正方形内
                if (not isPtInSquare(pt1, (pt_col, pt_row), size)) or (not isPtInSquare(pt2, (pt_col, pt_row), size)):
                    continue

                # 2 计算相似度最小的抓取标签
                metric = abs(grasp_row - pt_row) + abs(grasp_col - pt_col) + abs(grasp_angle_bin)
                if metric < min_metric:
                    min_metric = metric
                    grasp_correction = [grasp_row - pt_row, grasp_col - pt_col, grasp_angle_bin, grasp_width, grasp_depth]

        return grasp_correction
    
    def getCorrectGrasp(self, pt, size):
        """
        搜索最接近pt的抓取标签, 满足抓取线全部在正方形内
        相似度: 抓取点偏差+抓取角(弧度)偏差 dx+dy+dr  偏差最小且抓取线全部位于图像块内

        param:
            pt: 待修正的抓取点 (row, col)
            size: 正方形边长 int
        return: 
            抓取修正(除了抓取宽度和深度)
            [drow, dcol, dangle_bin, width, depth]  注意:dangle_bin
        """
        pt_row, pt_col = pt
        # 检索抓取点在正方形内的抓取配置
        crop_t = int(pt_row - size/2)
        crop_b = int(pt_row + size/2)
        crop_l = int(pt_col - size/2)
        crop_r = int(pt_col + size/2)
        segmask = np.zeros((self.height, self.width), dtype=np.uint8)
        segmask[crop_t:crop_b, crop_l:crop_r] = np.ones((size, size), dtype=np.uint8)

        grasp_candidate_pts = np.where(self.grasp_point_map_rot > 0)
        if grasp_candidate_pts[0].shape[0] == 0:
            return None
        grasp_candidate_pts = np.c_[grasp_candidate_pts[0], grasp_candidate_pts[1]] # shape=(n,2) row, col
        grasp_candidate_pts = np.array( [p for p in grasp_candidate_pts if np.any(segmask[p[0], p[1]] > 0)] )
        if grasp_candidate_pts.shape[0] == 0:
            return None

        min_metric = 1000
        grasp_correction = None     # 存储抓取修正
        for grasp_pt in grasp_candidate_pts:
            grasp_row, grasp_col = grasp_pt
            
            grasp_angle_bins = np.where(self.grasp_angle_map_rot[:, grasp_row, grasp_col] > 0)[0]
            for grasp_angle_bin in grasp_angle_bins:
                grasp_angle = grasp_angle_bin / self.angle_k * math.pi  # 抓取角 弧度
                grasp_width = self.grasp_width_map_rot[grasp_angle_bin, grasp_row, grasp_col]   # 抓取宽度 米
                grasp_depth = self.grasp_depth_map_rot[grasp_angle_bin, grasp_row, grasp_col]   # 抓取深度 米
                # 1 根据抓取线筛选
                # 计算两个抓取端点坐标
                pt1, pt2 = calcEndPoints((grasp_col, grasp_row), grasp_angle, grasp_width * RADIO)
                # 判断两个抓取端点是否在正方形内
                if (not isPtInSquare(pt1, (pt_col, pt_row), size)) or (not isPtInSquare(pt2, (pt_col, pt_row), size)):
                    continue

                # 2 计算相似度最小的抓取标签
                metric = abs(grasp_row - pt_row) + abs(grasp_col - pt_col) + abs(grasp_angle_bin)
                if metric < min_metric:
                    min_metric = metric
                    grasp_correction = [grasp_row - pt_row, grasp_col - pt_col, grasp_angle_bin, grasp_width, grasp_depth]

        return grasp_correction
    

    def getCorrectGrasp_1(self, pt, size):
        """
        搜索最接近pt的抓取标签, 满足抓取线全部在正方形内
        相似度: 抓取角(弧度)偏差 dx+dy+dr  偏差最小且抓取线全部位于图像块内

        param:
            pt: 待修正的抓取点 (row, col)
            size: 正方形边长 int
        return: 
            抓取修正(除了抓取宽度和深度)
            [success, dangle_bin, width, depth]  注意:dangle_bin
        """
        pt_row, pt_col = pt
        # 检索抓取点在正方形内的抓取配置
        crop_t = pt_row - 2
        crop_b = pt_row + 3
        crop_l = pt_col - 2
        crop_r = pt_col + 3
        segmask = np.zeros((self.height, self.width), dtype=np.uint8)
        segmask[crop_t:crop_b, crop_l:crop_r] = np.ones((5, 5), dtype=np.uint8)
        grasp_point_map_rot = self.grasp_point_map_rot * segmask

        grasp_candidate_pts = np.where(grasp_point_map_rot > 0)
        if grasp_candidate_pts[0].shape[0] == 0:
            return None

        grasp_candidate_pts = np.c_[grasp_candidate_pts[0], grasp_candidate_pts[1]]

        min_angle = 18
        grasp_correction = None     # 存储抓取修正
        for grasp_pt in grasp_candidate_pts:
            grasp_row, grasp_col = grasp_pt
            
            grasp_angle_bins = np.where(self.grasp_angle_map_rot[:, grasp_row, grasp_col] > 0)[0]
            for grasp_angle_bin in grasp_angle_bins:
                grasp_angle = grasp_angle_bin / self.angle_k * math.pi  # 抓取角 弧度
                grasp_width = self.grasp_width_map_rot[grasp_angle_bin, grasp_row, grasp_col]   # 抓取宽度 米
                grasp_depth = self.grasp_depth_map_rot[grasp_angle_bin, grasp_row, grasp_col]   # 抓取深度 米
                # 1 根据抓取线筛选
                # 计算两个抓取端点坐标
                pt1, pt2 = calcEndPoints((grasp_col, grasp_row), grasp_angle, grasp_width * RADIO)
                # 判断两个抓取端点是否在正方形内
                if (not isPtInSquare(pt1, (pt_col, pt_row), size)) or (not isPtInSquare(pt2, (pt_col, pt_row), size)):
                    continue

                # 2 计算相似度最小的抓取标签
                if grasp_angle_bin < min_angle:
                    min_angle = grasp_angle_bin
                    grasp_correction = [grasp_angle_bin, grasp_width, grasp_depth]
                    
        if grasp_correction is None:
            return None

        return grasp_correction


    def drawGrasps(self, img, interval=2):
        """
        绘制抓取配置
        """
        grasps = []
        rows, cols = np.where(self.grasp_point_map > 0)    # 抓取点
        for i in range(rows.shape[0]):
            if i % interval != 0:
                continue
            row, col = rows[i], cols[i]
            angle_bins = np.where(self.grasp_angle_map[:, row, col] > 0)[0]
            for angle_bin in angle_bins:
                angle = (angle_bin / self.angle_k) * math.pi
                width = self.grasp_width_map[angle_bin, row, col] * RADIO

                grasps.append([row, col, angle, width])

        # 绘制抓取
        img = drawGrasps(img, grasps, mode='line')

        return img

    def drawGraspsRot(self, img, interval=2):
        """
        绘制抓取配置
        """
        grasps = []
        rows, cols = np.where(self.grasp_point_map_rot > 0)    # 抓取点
        for i in range(rows.shape[0]):
            if i % interval != 0:
                continue
            row, col = rows[i], cols[i]
            angle_bins = np.where(self.grasp_angle_map_rot[:, row, col] > 0)[0]
            for angle_bin in angle_bins:
                angle = (angle_bin / self.angle_k) * math.pi
                width = self.grasp_width_map_rot[angle_bin, row, col] * RADIO

                grasps.append([row, col, angle, width])

        # 绘制抓取
        img = drawGrasps(img, grasps, mode='line')

        return img




if __name__ == "__main__":
    pt = (10, 10)
    angle = math.pi / 4
    width = 10
    pt1, pt2 = calcEndPoints(pt, angle, width)
    print(pt1, pt2)

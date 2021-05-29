"""
用于验证生成的抓取配置标注
随机放置物体，渲染图像，生成抓取配置，仿真机器人抓取

操作步骤：
1、运行后自动开始加载物体到仿真环境中。
2、按 1 开始渲染深度图。
3、等待渲染结束，按 2 开始计算抓取位姿，计算完毕后，自动开始抓取。
4、按 3 可重新加载物体。
"""

import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy as np
from simEnv import SimEnv
import panda_sim_grasp as panda_sim


GRASP_GAP = 0.005
GRASP_DEPTH = 0.005

def run():
    # 'E:/research/dataset/grasp/my_dense/mesh_database/egad_train_set'
    database_path = 'E:/research/dataset/grasp/my_dense/mesh_database/dex-net'      # 数据库路径

    cid = p.connect(p.GUI)  # 连接服务器
    env = SimEnv(p, database_path) # 初始化虚拟环境
    # 初始化panda机器人
    panda = panda_sim.PandaSimAuto(p, [0, -0.5, 0])

    GRASP_STATE = False
    grasp_config = {'x':0, 'y':0, 'z':0.05, 'angle':0, 'width':0.08}
    # x y z width的单位是m, angle的单位是弧度
    img_path = 'img/img_urdf'

    all_num = 0     # 预设抓取次数
    obj_nums = 20    # 每次加载的物体个数
    start_idx = 0   # 开始加载的物体id

    idx = start_idx
    while True:
        # 加载物体
        env.loadObjsInURDF(idx, obj_nums)
        idx += obj_nums

        while True:
            p.stepSimulation()
            time.sleep(1./240.)

            # 检测按键
            keys = p.getKeyboardEvents()
            if ord('1') in keys and keys[ord('1')]&p.KEY_WAS_TRIGGERED: 
                # 渲染图像
                env.renderURDFImage(save_path=img_path)

            if ord('2') in keys and keys[ord('2')]&p.KEY_WAS_TRIGGERED:
                # 计算抓取配置
                # graspFig = getGrasp(path=img_path)
                # print('抓取配置: ', graspFig)
                # if graspFig[4] == 0:
                #     print('>> 无法抓取!')
                #     continue
                # grasp_config['x'], grasp_config['y'], grasp_config['z'], grasp_config['angle'], grasp_config['width'] = graspFig
                GRASP_STATE = True

            if GRASP_STATE:
                # 机器人抓取
                if panda.step([grasp_config['x'], grasp_config['y'], grasp_config['z'] - GRASP_DEPTH], grasp_config['angle'], (grasp_config['width'])/2):
                    GRASP_STATE = False
                    all_num += 1
                    print('>> 抓取完毕: ', all_num)

            # 按3重置环境            
            if ord('3') in keys and keys[ord('3')]&p.KEY_WAS_TRIGGERED:
                env.removeObjsInURDF()
                break



if __name__ == "__main__":
    run()
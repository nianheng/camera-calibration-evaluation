import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import os
import numpy as np
import cv2 as cv

from calibration_verification import evaluate_calibration
from dual_quaternion import DualQuaternion

data_path = ('./data/chessboard/')

def parse_pose_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    poses = {}
    for child in root:
        time_key = child.tag
        data = child.find('data').text.strip().split()
        data = list(map(float, data))
        xyz = data[:3]
        quaternion = data[3:]
        poses[time_key] = {'xyz': xyz, 'quaternion': quaternion}
    return poses

def p_norm(pose):
    return pose / np.linalg.norm(pose)

def q_norm(quats):
    Quats = []
    for pose in quats:
        Quats.append(p_norm(pose))
    return np.array(Quats)


# 旋转矩阵->四元数
def matrix2quat(mat):
    rotation_mat = R.from_matrix(mat)
    return rotation_mat.as_quat()

# 旋转矩阵->四元数
def quat2matrix(quat):
    rot = R.from_quat(quat) # 顺序为 (x, y, z, w)
    return rot.as_matrix()

pose_data = parse_pose_xml(data_path + 'pose.xml')
w, h = 11, 8
checker_size = 35  #棋盘格大小,单位mm

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 阈值
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objp = objp * checker_size
objpoints = []  # 世界坐标系三维点
imgpoints = []  # 图像平面二维点
xyz = []  # arm位姿
quaternion = []  # arm四元数
i = 0
for time, pose in pose_data.items():
    if os.path.exists(data_path + time.split('_')[1] + f'.png'):
        quaternion.append([pose['quaternion'][0], pose['quaternion'][1], pose['quaternion'][2], pose['quaternion'][3]])
        xyz.append([1000 * pose['xyz'][0], 1000 * pose['xyz'][1], 1000 * pose['xyz'][2]])

        img = cv.imread(data_path + time.split('_')[1] + f'.png')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        u, v = img.shape[:2]
        # 找到棋盘格角点,SB函数鲁棒性更好,噪点忍受力高
        ret, corners = cv.findChessboardCorners(gray, (w, h), None)
        #存储
        if ret:
            # print(f"第{i + 1}张图片生成点阵")
            i = i + 1
            # 在原角点的基础上寻找亚像素角点
            cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # 追加进入世界三维点和平面二维点中
            objpoints.append(objp)
            imgpoints.append(corners)
            cv.drawChessboardCorners(img, (w, h), corners, ret)
            # cv.namedWindow('findCorners', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
            # cv.resizeWindow('findCorners', 640, 480)
            # cv.imshow('findCorners', img)
            # cv.waitKey(100)
cv.destroyAllWindows()
# 标定
print('正在计算')
# 标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("retval:", ret)
# 内参数矩阵
print("cameraMatrix内参矩阵:\n", camera_matrix)
# 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("distCoeffs畸变值:\n", np.array(dist_coeffs))
# 旋转向量  # 外参数
print("rvecs旋转向量外参:\n", np.array(rvecs))
# 平移向量  # 外参数
print("tvecs平移向量外参:\n", np.array(tvecs))
newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (u, v), 0, (u, v))
print('newcameramtx内参', newcameramtx)

# N*3*1旋转矢量
rvec = np.zeros((len(rvecs), 3, 1), np.double)
# N*3*1平移XYZ
tvec = np.zeros((len(tvecs), 3, 1), np.double)
# 获得新的标定板在相机坐标系下的位姿信息
for num_photo in range(len(xyz)):
    retval, rv, tv = cv.solvePnP(np.array(objpoints[num_photo]), np.array(imgpoints[num_photo]), camera_matrix,
                                 dist_coeffs)
    rvec[num_photo] = rv
    tvec[num_photo] = tv

R_bTg = []
t_bTg = []
R_target2cam = []
t_target2cam = []
xyz = np.array(xyz)
quaternion = np.array(quaternion)

#为各个参数赋值
for i in range(len(xyz)):
    # 处理R_bTg和t_bTg
    # 创建新的4*4的矩阵用来存放R和T拼接矩阵
    rtarray = np.zeros((4, 4), np.double)
    # 旋转向量转旋转矩阵,dst是是旋转矩阵,jacobian是雅可比矩阵
    # dst, jacobian = cv.Rodrigues(rxryrz[i])
    r = R.from_quat(quaternion[i])
    dst = r.as_matrix()

    # print('xyz:', xyz[i])
    # 存入旋转矩阵
    rtarray[:3, :3] = dst
    # 传入平移向量
    rtarray[:3, 3] = xyz[i]
    rtarray[3, 3] = 1
    # 放入用来传给calibrateHandEye的参数中
    R_bTg.append(dst)
    t_bTg.append(xyz[i])

    # 处理R_target2cam和t_target2cam
    # 获标定板在相机坐标系下的旋转矩阵,把旋转向量转成旋转矩阵
    dst1, jacobian1 = cv.Rodrigues(rvec[i])
    # 相机坐标系旋转矩阵转置
    R_target2cam.append(dst1)
    # 写入相机坐标系平移向量,平移向量需要乘原本的负的旋转矩阵
    t_target2cam.append(tvec[i])

# for i in range(len(t_bTg)):
#     print("t_bTg[" + str(i) + "]:", t_bTg[i], "t_target2cam[" + str(i) + "]:", t_target2cam[i].flatten())

# for i in range(len(t_bTg)):
#     print("R_bTg[" + str(i) + "]:", R_bTg[i], "R_target2cam[" + str(i) + "]:", R_target2cam[i].flatten())

# 核心方法,前面都是为了得到该方法的参数,获得转换矩阵
r_gTc, t_gTc = cv.calibrateHandEye(R_bTg, t_bTg, R_target2cam, t_target2cam, method=cv.CALIB_HAND_EYE_TSAI)
# 拼接出转换矩阵
rt = np.vstack((np.hstack((r_gTc, t_gTc)), np.array([0, 0, 0, 1])))
# results = np.zeros((3, 4, 4))
for i in range(len(t_bTg)):
    # print(str(i) + " ")
    base = np.column_stack((R_bTg[i], t_bTg[i]))
    base = np.row_stack((base, np.array([0, 0, 0, 1])))

    gripper = np.column_stack((R_target2cam[i], t_target2cam[i]))
    gripper = np.row_stack((gripper, np.array([0, 0, 0, 1])))

    result = np.matmul(np.matmul(base, rt), gripper)
    # result[i] = result
    # print(repr(result))
print('相机相对于末端的变换矩阵为：')
print(rt)
print("cameraMatrix内参矩阵:\n", camera_matrix)

# rotation_gTc = R.from_matrix(r_gTc)
# q_gTc = rotation_gTc.as_quat()  # 四元数表示 (x, y, z, w)
q_gTc = matrix2quat(r_gTc)  # 四元数表示 (x, y, z, w)

print('相机相对于末端的变换：')
print('位移 (t_x, t_y, t_z):', t_gTc.flatten())
print('四元数 (x, y, z, w):', q_gTc)

gTc = rt
print('gripper 下的 camera 位姿：')
print(gTc)

# for i in range(0, len(t_bTg)):
#     c_mat = np.hstack((R_target2cam[i], t_target2cam[i] / 100))
#     np.save(f'F:/work/paper/PE/CameraViewer-main/inputs/quick/debug/poses/{i}.npy', c_mat)

t_W_E = np.array(t_target2cam).squeeze()
q_W_E = np.array([matrix2quat(rot) for rot in R_target2cam])
# t_W_E = np.array(tvecs).squeeze()
# q_W_E = np.array([matrix2quat(cv.Rodrigues(rot)[0]) for rot in rvecs])

t_H_E = np.array(t_gTc).squeeze()
q_H_E = np.array(q_gTc)

# pose_B_H = np.hstack((xyz, q_norm(quaternion)))
# pose_W_E = np.hstack((t_W_E, q_norm(q_W_E)))
# pose_H_E = np.hstack((t_H_E, p_norm(q_H_E)))

pose_B_H = np.hstack((xyz, quaternion))
pose_W_E = np.hstack((t_W_E, q_W_E))
pose_H_E = np.hstack((t_H_E, q_H_E))

# pose_H_E[0] = pose_H_E[0] - 10
# pose_H_E[1] = pose_H_E[1] - 12
# pose_H_E[2] = pose_H_E[2] + 45

# dq_B_H_0 = DualQuaternion.from_pose_vector(pose_B_H[0])
# dq_W_E_0 = DualQuaternion.from_pose_vector(pose_W_E[0])
# dq_H_E = DualQuaternion.from_pose_vector(pose_H_E)
# dq_B_W_0 = dq_B_H_0 * dq_H_E * dq_W_E_0
# dq_B_W_0.normalize()

# dq_B_H_1 = DualQuaternion.from_pose_vector(pose_B_H[1])
# dq_W_E_1 = DualQuaternion.from_pose_vector(pose_W_E[1])
# dq_B_W_1 = dq_B_H_1 * dq_H_E * dq_W_E_1
# dq_B_W_1.normalize()

# dq_01 = dq_B_W_0 * dq_B_W_1.inverse()
# dq_00 = dq_B_W_0 * dq_B_W_0.inverse()

(rmse_position, rmse_orientation) = evaluate_calibration(pose_B_H, pose_W_E, pose_H_E)

print(f"rmse_position: {rmse_position}")
print(f"rmse_orientation: {rmse_orientation}")

dq_B_H_0 = DualQuaternion.from_pose_vector(pose_B_H[0])
dq_W_E_0 = DualQuaternion.from_pose_vector(pose_W_E[0]).inverse()
dq_H_E = DualQuaternion.from_pose_vector(pose_H_E)
dq_B_W_0 = dq_B_H_0 * dq_H_E * dq_W_E_0.inverse()
dq_B_W_0.normalize()


for i in range(0, 6):
    pos_vec = t_target2cam[i].flatten() / 100 - [0, 0, 5]
    rot_vec = pos_vec + R_target2cam[i].dot(np.array([0, 0, 1]))

    dq_B_H_i = DualQuaternion.from_pose_vector(pose_B_H[i])
    dq_W_E_estimate = dq_B_W_0.inverse() * dq_B_H_i * dq_H_E
    dq_W_E_estimate = dq_W_E_estimate.inverse()

    pose_W_E_estimate = dq_W_E_estimate.to_pose().T
    pos_vec_estimate = pose_W_E_estimate[:3] / 100 - [0, 0, 5]
    rot_vec_estimate = pos_vec_estimate + quat2matrix(pose_W_E_estimate[3:]).dot(np.array([0, 0, 1]))

    # rot_vec = pos_vec + np.array([1, 1, 1]).dot(R_target2cam[i])
    with open(f'F:/work/paper/PE/CameraViewer-main/inputs/quick/debug1/poses/{i}.txt', 'w') as file:
        file.write(f'{pos_vec[0]} {pos_vec[1]} {pos_vec[2]}\n{rot_vec[0]} {rot_vec[1]} {rot_vec[2]}\n0 0 1')
    with open(f'F:/work/paper/PE/CameraViewer-main/inputs/quick/debug1/poses/{6+i}.txt', 'w') as file:
        file.write(f'{pos_vec_estimate[0]} {pos_vec_estimate[1]} {pos_vec_estimate[2]}\n{rot_vec_estimate[0]} {rot_vec_estimate[1]} {rot_vec_estimate[2]}\n0 0 1')

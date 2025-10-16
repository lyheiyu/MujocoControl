import cv2
import mediapipe as mp
import mujoco
import glfw
import numpy as np  # 新增，用于数组操作

# 初始化 Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 初始化 GLFW 窗口 (用于 MuJoCo 渲染)
if not glfw.init():
    raise Exception("无法初始化 GLFW")

# 创建一个 640x480 的窗口用于渲染 MuJoCo 模型
window = glfw.create_window(640, 480, "MuJoCo Viewer", None, None)
glfw.make_context_current(window)

# 加载 MuJoCo 模型
model = mujoco.MjModel.from_xml_path('C:\D\PC desktop\Robot\mujoco_menagerie-main\mujoco_menagerie-main\shadow_hand/left_hand.xml')
data = mujoco.MjData(model)

# 创建渲染场景和上下文
scene = mujoco.MjvScene(model, maxgeom=1000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# 初始化 MjvOption，用于场景更新
opt = mujoco.MjvOption()

# 初始化摄像机
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # 设置为自由视角
# 设置摄像机的位置，使其更接近手部
cam.lookat = np.array([0, 0, 0])  # 设定摄像机朝向的目标位置
cam.distance = 0.1  # 调整摄像机到目标位置的距离，数值越小，物体越大
cam.elevation = -90  # 设定摄像机角度，使其俯视或平视手部
cam.azimuth = 90  # 调整方位角度，改变摄像机的横向角度


# 初始化 perturb，用于操控
pert = mujoco.MjvPerturb()

# 获取所有关节的 actuator 索引
actuators = {
    'wrist_x': model.actuator('lh_A_WRJ1').id,
    'wrist_y': model.actuator('lh_A_WRJ2').id,
    'thumb_base': model.actuator('lh_A_THJ5').id,
    'thumb_proximal': model.actuator('lh_A_THJ4').id,
    'thumb_hub': model.actuator('lh_A_THJ3').id,
    'thumb_middle': model.actuator('lh_A_THJ2').id,
    'thumb_distal': model.actuator('lh_A_THJ1').id,
    'index_knuckle': model.actuator('lh_A_FFJ4').id,
    'index_proximal': model.actuator('lh_A_FFJ3').id,
    'index_distal': model.actuator('lh_A_FFJ0').id,
    'middle_knuckle': model.actuator('lh_A_MFJ4').id,
    'middle_proximal': model.actuator('lh_A_MFJ3').id,
    'middle_distal': model.actuator('lh_A_MFJ0').id,
    'ring_knuckle': model.actuator('lh_A_RFJ4').id,
    'ring_proximal': model.actuator('lh_A_RFJ3').id,
    'ring_distal': model.actuator('lh_A_RFJ0').id,
    'little_knuckle': model.actuator('lh_A_LFJ4').id,
    'little_proximal': model.actuator('lh_A_LFJ3').id,
    'little_distal': model.actuator('lh_A_LFJ0').id
}

# 打开摄像头
cap = cv2.VideoCapture(0)

# 函数：计算两个3D点之间的向量
def compute_vector(point1, point2):
    return np.array([point2.x - point1.x, point2.y - point1.y, point2.z - point1.z])

# 函数：计算两个向量之间的夹角
def compute_angle(v1, v2):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2) / (v1_norm * v2_norm)
    return np.arccos(cos_theta)

while cap.isOpened() and not glfw.window_should_close(window):
    ret, frame = cap.read()
    if not ret:
        break

    # 镜像翻转摄像头输入
    frame = cv2.flip(frame, 1)

    # 转换为 RGB 图像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 使用 Mediapipe 处理手部关键点
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 绘制手部关键点
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 控制拇指的关节
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            # thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
            thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
            thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # 计算拇指各关节的角度
            v0 =compute_vector(wrist,thumb_cmc)
            v1 = compute_vector(thumb_cmc, thumb_mcp)
            v2 = compute_vector(thumb_mcp, thumb_ip)
            v3 = compute_vector(thumb_ip, thumb_tip)
            thumb_base_angle = compute_angle(v0, v1)
            thumb_second_angle= compute_angle(v2,v1)
            thumb_third_angle = compute_angle(v2, v3)

            # thumb_proximal_angle = compute_angle(v2, v3)

            # 将计算出的角度映射到 MuJoCo 的 actuator 控制
            data.ctrl[actuators['thumb_base']] = thumb_base_angle
            # data.ctrl[actuators['thumb_proximal']] = thumb_base_angle-0.5
            data.ctrl[actuators['thumb_middle']] = thumb_second_angle
            data.ctrl[actuators['thumb_distal']] = thumb_third_angle
            # 控制食指的关节

            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_knuckle_angle = compute_angle(compute_vector(wrist, index_mcp), compute_vector(index_mcp, index_pip))
            vi0= compute_vector(wrist,index_mcp)
            vi1 = compute_vector(index_mcp, index_pip)
            vi2 = compute_vector(index_pip, index_dip)
            vi3 = compute_vector(index_dip, index_tip)
            index_1_angle = compute_angle(vi0, vi3)
            index_2_angle = compute_angle(vi0, vi1)
            index_3_angle = compute_angle(vi1, vi2)
            index_proximal_angle = compute_angle(vi2, vi3)

            # 将计算出的角度映射到 MuJoCo 的 actuator 控制
            print(index_knuckle_angle)
            data.ctrl[actuators['index_knuckle']] = index_knuckle_angle-0.5
            data.ctrl[actuators['index_proximal']] = index_2_angle
            data.ctrl[actuators['index_distal']] = index_3_angle
            # 你可以重复以上步骤，控制中指、无名指和小指
            MIDDLE_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            MIDDLE_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            MIDDLE_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
            MIDDLE_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_knuckle_angle = compute_angle(compute_vector(wrist, MIDDLE_mcp), compute_vector(index_mcp, MIDDLE_pip))
            vmx= compute_vector(index_mcp,MIDDLE_mcp)
            vm0 = compute_vector(wrist, MIDDLE_mcp)
            vm1 = compute_vector(MIDDLE_mcp, MIDDLE_pip)
            vm2 = compute_vector(MIDDLE_pip, MIDDLE_dip)
            vm3 = compute_vector(MIDDLE_dip, MIDDLE_tip)
            MIDDLE_1_angle = compute_angle(vmx, vm3)
            MIDDLE_2_angle = compute_angle(vm0, vm1)
            MIDDLE_3_angle = compute_angle(vm1, vm2)
            MIDDLE_proximal_angle = compute_angle(vm2, vm3)

            # 将计算出的角度映射到 MuJoCo 的 actuator 控制
            data.ctrl[actuators['middle_knuckle']] = middle_knuckle_angle-0.5
            data.ctrl[actuators['middle_proximal']] = MIDDLE_2_angle
            data.ctrl[actuators['middle_distal']] = MIDDLE_3_angle
            # ring finger

            RING_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            RING_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
            RING_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
            RING_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_knuckle_angle = compute_angle(compute_vector(wrist, RING_mcp),
                                                 compute_vector(index_mcp, RING_pip))

            vrx = compute_vector(RING_mcp, MIDDLE_mcp)
            vr0 = compute_vector(wrist, RING_mcp)
            vr1 = compute_vector(RING_mcp, RING_pip)
            vr2 = compute_vector(RING_pip, RING_dip)
            vr3 = compute_vector(RING_dip, RING_tip)
            RING_1_angle = compute_angle(vrx, vr3)
            RING_2_angle = compute_angle(vr0, vr1)
            RING_3_angle = compute_angle(vr1, vr2)
            RING_proximal_angle = compute_angle(vr2, vr3)

            # 将计算出的角度映射到 MuJoCo 的 actuator 控制
            data.ctrl[actuators['ring_knuckle']] = 1-ring_knuckle_angle
            data.ctrl[actuators['ring_proximal']] = RING_2_angle
            data.ctrl[actuators['ring_distal']] = RING_3_angle

            # little finger

            Little_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            Little_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
            Little_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
            Little_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            little_knuckle_angle = compute_angle(compute_vector(wrist, Little_mcp),
                                               compute_vector(index_mcp, Little_pip))
            vlx = compute_vector(RING_mcp, Little_mcp)
            vl0 = compute_vector(wrist, Little_mcp)
            vl1 = compute_vector(Little_mcp, Little_pip)
            vl2 = compute_vector(Little_pip, Little_dip)
            vl3 = compute_vector(Little_dip, Little_tip)
            Little_1_angle = compute_angle(vlx, vl3)
            Little_2_angle = compute_angle(vl0, vl1)
            Little_3_angle = compute_angle(vl1, vl2)
            Little_proximal_angle = compute_angle(vl2, vl3)

            # 将计算出的角度映射到 MuJoCo 的 actuator 控制
            data.ctrl[actuators['little_knuckle']] =1- little_knuckle_angle
            data.ctrl[actuators['little_proximal']] = Little_2_angle
            data.ctrl[actuators['little_distal']] = Little_3_angle
    # 更新仿真
    mujoco.mj_step(model, data)

    # 更新并渲染 MuJoCo 模型
    viewport = mujoco.MjrRect(0, 0, 640, 480)
    mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)

    # 显示摄像头图像
    cv2.imshow("Hand Tracking", frame)

    # 检查是否按下了 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 更新窗口并处理事件
    glfw.swap_buffers(window)
    glfw.poll_events()

# 释放资源
cap.release()
cv2.destroyAllWindows()
glfw.terminate()

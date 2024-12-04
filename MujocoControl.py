import cv2
import mediapipe as mp
import mujoco
import glfw

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
model = mujoco.MjModel.from_xml_path('C:/Users/35389/Desktop/Robot/mujoco_menagerie-main/mujoco_menagerie-main/shadow_hand/left_hand.xml')
data = mujoco.MjData(model)

# 创建渲染场景和上下文
scene = mujoco.MjvScene(model, maxgeom=1000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# 初始化 MjvOption，用于场景更新
opt = mujoco.MjvOption()

# 初始化摄像机
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # 设置为自由视角

# 初始化 perturb，用于操控
pert = mujoco.MjvPerturb()

# 获取所有关节的 actuator 索引
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

# 关节的位置获取使用有效的关节名称
previous_positions = {
    'wrist_x': data.qpos[model.joint('lh_WRJ1').qposadr],
    'wrist_y': data.qpos[model.joint('lh_WRJ2').qposadr],
    'thumb_base': data.qpos[model.joint('lh_THJ5').qposadr],
    'thumb_proximal': data.qpos[model.joint('lh_THJ4').qposadr],
    'thumb_hub': data.qpos[model.joint('lh_THJ3').qposadr],
    'thumb_middle': data.qpos[model.joint('lh_THJ2').qposadr],
    'thumb_distal': data.qpos[model.joint('lh_THJ1').qposadr],
    'index_knuckle': data.qpos[model.joint('lh_FFJ4').qposadr],
    'index_proximal': data.qpos[model.joint('lh_FFJ3').qposadr],
    'index_middle': data.qpos[model.joint('lh_FFJ2').qposadr],
    'index_distal': data.qpos[model.joint('lh_FFJ1').qposadr],
    'middle_knuckle': data.qpos[model.joint('lh_MFJ4').qposadr],
    'middle_proximal': data.qpos[model.joint('lh_MFJ3').qposadr],
    'middle_middle': data.qpos[model.joint('lh_MFJ2').qposadr],
    'middle_distal': data.qpos[model.joint('lh_MFJ1').qposadr],
    'ring_knuckle': data.qpos[model.joint('lh_RFJ4').qposadr],
    'ring_proximal': data.qpos[model.joint('lh_RFJ3').qposadr],
    'ring_middle': data.qpos[model.joint('lh_RFJ2').qposadr],
    'ring_distal': data.qpos[model.joint('lh_RFJ1').qposadr],
    'little_knuckle': data.qpos[model.joint('lh_LFJ4').qposadr],
    'little_proximal': data.qpos[model.joint('lh_LFJ3').qposadr],
    'little_middle': data.qpos[model.joint('lh_LFJ2').qposadr],
    'little_distal': data.qpos[model.joint('lh_LFJ1').qposadr]
}



# 打开摄像头
cap = cv2.VideoCapture(0)

# 记录之前的位置以测量差异
previous_positions = {joint: 0 for joint in actuators}

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

            # 获取手部各关节的位置
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
            thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
            ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
            ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
            little_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            little_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
            little_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]

            # 控制手腕
            wrist_x_control = (wrist.x - 0.2) * 20
            wrist_x_control = max(min(wrist_x_control, 4), -4)
            data.ctrl[actuators['wrist_x']] = wrist_x_control
            wrist_y_control = (wrist.y - 0.5) * 10
            wrist_y_control = max(min(wrist_y_control, 4), -4)
            data.ctrl[actuators['wrist_y']] = wrist_y_control

            # 控制拇指
            data.ctrl[actuators['thumb_base']] = (thumb_cmc.x - 0.5) * 10
            data.ctrl[actuators['thumb_proximal']] = (thumb_mcp.x - 0.5) * 10
            data.ctrl[actuators['thumb_middle']] = (thumb_ip.x - 0.5) * 10

            # 控制食指
            data.ctrl[actuators['index_knuckle']] = (index_mcp.x - 0.5) * 10
            data.ctrl[actuators['index_proximal']] = (index_pip.x - 0.5) * 10
            data.ctrl[actuators['index_distal']] = (index_dip.x - 0.5) * 10

            # 控制中指
            data.ctrl[actuators['middle_knuckle']] = (middle_mcp.x - 0.5) * 10
            data.ctrl[actuators['middle_proximal']] = (middle_pip.x - 0.5) * 10
            data.ctrl[actuators['middle_distal']] = (middle_dip.x - 0.5) * 10

            # 控制无名指
            data.ctrl[actuators['ring_knuckle']] = (ring_mcp.x - 0.5) * 10
            data.ctrl[actuators['ring_proximal']] = (ring_pip.x - 0.5) * 10
            data.ctrl[actuators['ring_distal']] = (ring_dip.x - 0.5) * 10

            # 控制小指
            data.ctrl[actuators['little_knuckle']] = (little_mcp.x - 0.5) * 10
            data.ctrl[actuators['little_proximal']] = (little_pip.x - 0.5) * 10
            data.ctrl[actuators['little_distal']] = (little_dip.x - 0.5) * 10

            # 更新仿真
            mujoco.mj_step(model, data)

            # 获取当前的关节位置并计算差值
            # 正确获取关节的名称
            for joint, actuator_id in actuators.items():
                if joint in previous_positions:
                    # 直接使用映射的关节名称获取位置
                    position = previous_positions[joint]
                    print(f"{joint} Position Difference: {position - previous_positions[joint]}")
                    previous_positions[joint] = position

    # 更新并渲染 MuJoCo 模型
    mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, 640, 480), scene, context)

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

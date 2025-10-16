import serial, math, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ==== 串口配置 ====
PORT = "COM4"     # ← 改成你的端口
BAUD = 115200
# ==================

def euler_to_R(roll_deg, pitch_deg, yaw_deg):
    # ZYX顺序：Rz(yaw)*Ry(pitch)*Rx(roll)
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    sr, cr = math.sin(r), math.cos(r)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    Rz = np.array([[cy, -sy, 0],[sy, cy, 0],[0, 0, 1]])
    Ry = np.array([[cp, 0, sp],[0, 1, 0],[-sp, 0, cp]])
    Rx = np.array([[1, 0, 0],[0, cr, -sr],[0, sr, cr]])
    return Rz @ Ry @ Rx

def parse_line(line: str):
    # 期望：t,roll,pitch,yaw0,ax,ay,az,gx,gy,gz
    parts = line.strip().split(',')
    if len(parts) < 4: return None
    try:
        return float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None

# 串口
ser = serial.Serial(PORT, BAUD, timeout=0.05)
try:
    ser.setDTR(False); ser.setRTS(False)
except Exception:
    pass
time.sleep(0.3)
ser.reset_input_buffer()
print("Opened", ser.name)

# 3D 画布
plt.style.use("dark_background")
fig = plt.figure("ESP32 + MPU6050 3D Pose", figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.2,1.2]); ax.set_ylim([-1.2,1.2]); ax.set_zlim([-1.2,1.2])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.view_init(elev=25, azim=45)

# 立方体（边长1，机体系），3x8
cube_verts = np.array([
    [-0.5,-0.5,-0.5],[ 0.5,-0.5,-0.5],[ 0.5, 0.5,-0.5],[-0.5, 0.5,-0.5],
    [-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5],
]).T
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
cube_lines = [ax.plot([], [], [], lw=2)[0] for _ in edges]

# 坐标轴
axis_len = 0.8
axis_lines = [ax.plot([], [], [], lw=3, color=c)[0] for c in ('r','g','b')]

txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
cur_rpy = [0.0, 0.0, 0.0]

def update(_):
    global cur_rpy
    # 尽量拿到最新一帧
    for _ in range(80):
        line = ser.readline().decode(errors="ignore")
        print(line)
        if not line: break
        data = parse_line(line)
        if data: cur_rpy = list(data)

    roll, pitch, yaw = cur_rpy
    R = euler_to_R(roll, pitch, yaw)
    cube_world = R @ cube_verts

    # 正确绘制每条边：分别取 x/y/z 两个端点
    for line, (i, j) in zip(cube_lines, edges):
        xs = [cube_world[0, i], cube_world[0, j]]
        ys = [cube_world[1, i], cube_world[1, j]]
        zs = [cube_world[2, i], cube_world[2, j]]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)

    # 坐标轴
    origin = np.zeros((3,))
    ex = (R @ np.array([axis_len,0,0])).flatten()
    ey = (R @ np.array([0,axis_len,0])).flatten()
    ez = (R @ np.array([0,0,axis_len])).flatten()
    for line, end in zip(axis_lines, [ex, ey, ez]):
        line.set_data([origin[0], end[0]], [origin[1], end[1]])
        line.set_3d_properties([origin[2], end[2]])

    txt.set_text(f"Roll: {roll:6.1f}°  Pitch: {pitch:6.1f}°  Yaw0: {yaw:6.1f}°")
    return cube_lines + axis_lines + [txt]

ani = FuncAnimation(fig, update, interval=150, blit=False, save_count=100, cache_frame_data=False)
plt.show()

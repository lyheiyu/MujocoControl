# ConnectToESP32.py
import serial, threading, collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PORT = "COM4"
BAUD = 115200
HIST = 300  # 显示最近 N 个点

# ------- 串口读取线程，把有效数据喂进队列 -------
q = collections.deque(maxlen=HIST)

def reader():
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        ser.reset_input_buffer()
        while True:
            raw = ser.readline()
            if not raw:
                continue
            try:
                s = raw.decode("utf-8", errors="ignore").strip()
            except:
                continue
            if not s or s.startswith("t,"):   # 跳过表头
                continue
            parts = s.split(",")
            # 期待格式: t,roll,pitch,yaw,ax,ay,az,gx,gy,gz
            if len(parts) < 4:
                continue
            try:
                roll  = float(parts[1])
                pitch = float(parts[2])
                yaw   = float(parts[3])
            except ValueError:
                continue
            q.append((roll, pitch, yaw))

thr = threading.Thread(target=reader, daemon=True)
thr.start()

# ------- 画图 -------
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8,4))
ax.set_title("ESP32 + MPU6050 (Serial)")
ax.set_xlabel("samples")
ax.set_ylabel("deg")
ax.set_ylim(-180, 180)

x = list(range(HIST))
roll_hist  = collections.deque([0.0]*HIST, maxlen=HIST)
pitch_hist = collections.deque([0.0]*HIST, maxlen=HIST)
yaw_hist   = collections.deque([0.0]*HIST, maxlen=HIST)

lr, = ax.plot(x, roll_hist,  label="Roll (deg)")
lp, = ax.plot(x, pitch_hist, label="Pitch (deg)")
ly, = ax.plot(x, yaw_hist,   label="Yaw (deg)")
ax.legend(loc="upper right")

def update(frame):
    # 把队列里的所有新数据吃掉
    while q:
        r, p, y = q.popleft()
        roll_hist.append(r)
        pitch_hist.append(p)
        yaw_hist.append(y)
    lr.set_ydata(roll_hist)
    lp.set_ydata(pitch_hist)
    ly.set_ydata(yaw_hist)
    return lr, lp, ly

ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
plt.show()

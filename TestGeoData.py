import numpy as np
import matplotlib.pyplot as plt

# 模拟NDVI数据（-1到1之间）
ndvi = np.random.uniform(-1, 1, (100, 100))

# 模拟坡度数据（0到45度）
slope = np.random.uniform(0, 45, (100, 100))

# 模拟DEM数据（0到2000米）
dem = np.random.uniform(0, 2000, (100, 100))

# 可视化每个数据
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ['NDVI', 'Slope (degrees)', 'DEM (meters)']
data = [ndvi, slope, dem]

for i, ax in enumerate(axes):
    im = ax.imshow(data[i], cmap='viridis')
    ax.set_title(titles[i])
    ax.axis('off')
    fig.colorbar(im, ax=ax, orientation='vertical')

plt.tight_layout()
plt.show()

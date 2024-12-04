import numpy as np
from stl import mesh

# 读取STL文件
your_mesh = mesh.Mesh.from_file('trapezoid.stl')

# 提取所有的顶点数据
points = your_mesh.vectors.reshape(-1, 3)

# 去重顶点（如果需要）
unique_points = np.unique(points, axis=0)

# 保存为点云
np.savetxt('point_cloud.txt', unique_points)


def generate_mujoco_composite(points, output_file):
    with open(output_file, 'w') as f:
        f.write('<mujoco>\n')
        f.write('  <worldbody>\n')
        f.write('    <body>\n')

        # Composite 定义，使用点云生成柔性体
        f.write('      <composite type="particle" count="{}" dim="3" spacing="0.1 0.1 0.1">\n'.format(len(points)))

        # 写入点云作为顶点
        f.write('        <vertex>\n')
        for point in points:
            f.write(f'          {point[0]} {point[1]} {point[2]}\n')
        f.write('        </vertex>\n')

        # 定义连接和几何元素
        f.write('        <geom type="sphere" size="0.01"/>\n')

        # 自动生成的 bodies 的关节、皮肤等
        f.write('        <joint type="slide"/>\n')
        f.write('        <skin rgba="0.8 0.2 0.2 1"/>\n')
        f.write('      </composite>\n')

        f.write('    </body>\n')
        f.write('  </worldbody>\n')
        f.write('</mujoco>\n')


# 将点云生成的 composite 导出为 MuJoCo XML 文件
generate_mujoco_composite(unique_points, 'generated_composite.xml')

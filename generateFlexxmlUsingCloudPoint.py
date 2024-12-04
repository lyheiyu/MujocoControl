import numpy as np
from stl import mesh

# 读取STL文件
your_mesh = mesh.Mesh.from_file('trapezoid.stl')

# 提取所有的顶点数据并去重
points = your_mesh.vectors.reshape(-1, 3)
unique_points = np.unique(points, axis=0)


# 定义函数生成MuJoCo XML
def generate_mujoco_xml(point_data, output_file):
    with open(output_file, 'w') as f:
        f.write('<mujoco>\n')
        f.write('  <worldbody>\n')
        f.write('    <body name="softbody">\n')
        f.write('      <joint type="free"/>\n')
        f.write('      <geom type="mesh" mesh="custom_mesh" pos="0 0 0" rgba="0 .7 .7 1"/>\n')

        # 写入柔性体定义
        f.write('      <deformable>\n')
        f.write('        <flex name="flex_body" dim="3" mass="5" radius="0.05"\n')

        # 将顶点作为数组写入 vertex 属性
        f.write('         vertex="')
        for point in point_data:
            f.write(f'{point[0]} {point[1]} {point[2]} ')
        f.write('"/>\n')

        # 结束flex和deformable部分
        f.write('        </flex>\n')
        f.write('      </deformable>\n')
        f.write('    </body>\n')
        f.write('  </worldbody>\n')
        f.write('</mujoco>\n')


# 生成MuJoCo XML文件
generate_mujoco_xml(unique_points, 'generated_mujoco_flex.xml')

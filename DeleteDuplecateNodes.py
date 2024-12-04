import gmsh

# 初始化 Gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

# 导入 STEP 文件
step_file = "C:/Users/35389/Desktop/Robot/mujoco/Flex body/bunny_simple/softbox/Dshape.step"
gmsh.model.occ.importShapes(step_file)
gmsh.model.occ.synchronize()

# 清除重复面并确保几何体闭合
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.fragment(gmsh.model.getEntities(), [])
gmsh.model.occ.synchronize()

# 获取所有的面实体并创建闭合的体积
entities = gmsh.model.getEntities(dim=2)
surfaces = [entity[1] for entity in entities]
if surfaces:
    surface_loop = gmsh.model.occ.addSurfaceLoop(surfaces)
    gmsh.model.occ.addVolume([surface_loop])

# 同步并移除重复节点
gmsh.model.occ.synchronize()
gmsh.model.mesh.removeDuplicateNodes()

# 设置网格划分尺寸
element_size = 500
max_element_size = 1000
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_element_size)

# 生成三维四面体网格
try:
    gmsh.model.mesh.generate(3)
except Exception as e:
    print(f"Mesh generation failed: {str(e)}")

# 保存优化后的 .msh 文件
output_msh_file = "C:/Users/35389/Desktop/Robot/mujoco/Flex body/bunny_simple/softbox/scaled_Dshape.msh"
gmsh.write(output_msh_file)

gmsh.finalize()

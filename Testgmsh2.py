import gmsh
import sys

# Initialize Gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)  # Enable terminal messages for debugging

# Create a new model
gmsh.model.add("high_density_tetrahedral_mesh")

# Import the STEP file, replace with your STEP file path if needed
step_file = "C:/Users/35389/Desktop/Robot/mujoco/Flex body/bunny_simple/Trapezoid3.step"
gmsh.model.occ.importShapes(step_file)

# Synchronize the geometry
gmsh.model.occ.synchronize()

# Use fragment to ensure closed geometry and fix any overlapping surfaces
gmsh.model.occ.fragment(gmsh.model.getEntities(), [])
gmsh.model.occ.synchronize()

# Retrieve all surfaces to create a volume
surfaces = [entity[1] for entity in gmsh.model.getEntities(2)]

# Create a Surface Loop and then a Volume
if surfaces:
    surface_loop = gmsh.model.occ.addSurfaceLoop(surfaces)
    gmsh.model.occ.addVolume([surface_loop])

# Synchronize again after creating the volume
gmsh.model.occ.synchronize()

# Set mesh density (smaller numbers create a denser mesh)
element_size_min = 500  # Adjust to increase node density
element_size_max = 500
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size_min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size_max)

# Set the output version to 2 ASCII
gmsh.option.setNumber("Mesh.MshFileVersion", 2)

# Generate only the 3D tetrahedral mesh
gmsh.model.mesh.generate(3)

# Export the mesh in .msh format for MuJoCo in the same directory as the STEP file
output_msh_file = "C:/Users/35389/Desktop/Robot/mujoco/Flex body/bunny_simple/high_density_tetrahedral_mesh.msh"
gmsh.write(output_msh_file)

# Optional: Visualize the mesh
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

# Finalize Gmsh
gmsh.finalize()

print("Mesh generation completed and saved as 4-node tetrahedrons.")

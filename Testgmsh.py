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

# Get all surface entities to define the surface loop and volume
entities = gmsh.model.getEntities(dim=2)
surfaces = [entity[1] for entity in entities]

# Create a Surface Loop and Volume for the 3D mesh
if surfaces:
    surface_loop = gmsh.model.occ.addSurfaceLoop(surfaces)
    gmsh.model.occ.addVolume([surface_loop])

# Synchronize again
gmsh.model.occ.synchronize()

# Set mesh density (smaller numbers create a denser mesh)
element_size_min = 200 # Adjust to increase node density
element_size_max = 500
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size_min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size_max)

# Generate the 3D tetrahedral mesh
try:
    gmsh.model.mesh.generate(3)
except Exception as e:
    print(f"Mesh generation failed: {str(e)}")

# Export the mesh in .msh format for MuJoCo
output_msh_file = "high_density_tetrahedral_mesh.msh"
gmsh.write(output_msh_file)

# Optional: Visualize the mesh
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

# Finalize Gmsh
gmsh.finalize()

# print hello

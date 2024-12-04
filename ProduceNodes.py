import re

def scale_msh_nodes(input_file, output_file, scale_factor=0.001):
    # Read the content of the original file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # To store modified lines
    scaled_lines = []
    in_nodes_section = False

    for line in lines:
        if line.startswith("$Nodes"):
            in_nodes_section = True
            scaled_lines.append(line)
        elif line.startswith("$EndNodes"):
            in_nodes_section = False
            scaled_lines.append(line)
        elif in_nodes_section:
            # If in the nodes section, scale the node coordinates
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    node_id = int(parts[0])
                    x, y, z = map(float, parts[1:4])
                    # Scale coordinates
                    x *= scale_factor
                    y *= scale_factor
                    z *= scale_factor
                    scaled_line = f"{node_id} {x} {y} {z}\n"
                    scaled_lines.append(scaled_line)
                except ValueError:
                    scaled_lines.append(line)
            else:
                scaled_lines.append(line)
        else:
            # If not in nodes section, keep line as is
            scaled_lines.append(line)

    # Write the scaled nodes to the output file
    with open(output_file, 'w') as file:
        file.writelines(scaled_lines)

    print(f"File '{output_file}' created with scaled node coordinates.")

# Example usage
input_file = "C:/Users/35389/Desktop/Robot/mujoco/Flex body/bunny_simple/softbox/Dshape2.msh"
output_file = "C:/Users/35389/Desktop/Robot/mujoco/Flex body/bunny_simple/softbox/scaled_Dshape.msh"
scale_msh_nodes(input_file, output_file)

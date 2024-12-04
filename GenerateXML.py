def generate_joints(num_joints, joint_type="hinge", pos="0 0 0", axis="0 0 1", size="0.0002"):
    joints = ""
    for i in range(1, num_joints + 1):
        joint = f'<joint name="joint {i}" type="{joint_type}" pos="{pos}" axis="{axis}" size="{size}"/>\n' \
                f'<geom type="box" size="0.001 0.001 0.001" />\n'

        joints += joint
    return joints

# Example: Generate 100 joints
num_joints = 200
joints_xml = generate_joints(num_joints)

# Write the output to a file (optional)
with open("joints.xml", "w") as f:
    f.write(joints_xml)

print(joints_xml)

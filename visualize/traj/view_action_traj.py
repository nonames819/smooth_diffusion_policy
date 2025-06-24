import os, h5py, sys
from cnom_visualization import visualize_traj
import pytorch3d.transforms as pt
import numpy as np

file_path = "/home/caohaidong/code/awe/robomimic/datasets_test/square/ph/low_dim.hdf5"
# key = "/data/demo_1/cs_info/world_space_action"
# key = "/data/demo_1/cs_info/eef_pose"
# key = "/data/demo_1/actions"
key = "/data/demo_1/actions"


f = h5py.File(file_path, 'r')
output_dir = os.path.join(os.path.dirname(file_path), 'info')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pos = []
if key.endswith('world_space_action'):
    print("Using world space action data")
    output_path = os.path.join(os.path.dirname(file_path), 'info/traj_action_from_pose_world.png')
    pose = f[key][:]
    pos = pose[:, :3, 3]
    visualize_traj(pos[:, 0], pos[:, 1], pos[:, 2], output_path)
elif key.endswith('eef_pose'):
    print("Using end-effector pose data")
    output_path = os.path.join(os.path.dirname(file_path), 'info/traj_action_from_pose.png')
    pose = f[key][:]
    pos = pose[:, :3, 3]
    visualize_traj(pos[:, 0], pos[:, 1], pos[:, 2], output_path)
elif key.endswith('actions'):
    print("Using action data")
    output_path = os.path.join(os.path.dirname(file_path), 'info/traj_action_rel.png')
    actions = f[key][:]
    pos_rel = actions[:, :3]
    pos = np.zeros_like(pos_rel)
    # pos[0,:] = [-0.56, 0, 0.912]
    print(f"pos[0]: {pos[0]}")
    for p in range(1,len(pos)):
        pos[p,:] = pos_rel[p-1,:]*0.05+pos[p-1,:]
        # print(f"pos[{p}]: {pos[p]}")
    visualize_traj(pos[:, 0],pos[:, 1],pos[:, 2], output_path)
    # pos = actions[:, :3]
    # for p in range(1, len(pos)):
    #     pos[p, :] = pos[p-1, :] + pos[p, :]
    #     # print(f"pos[{p}]: {pos[p]}")
elif key.endswith('actions_abs'):
    print("Using absolute action data")
    output_path = os.path.join(os.path.dirname(file_path), 'info/traj_action_abs.png')
    actions = f[key][:]
    pos = actions[:, :3]
    visualize_traj(pos[:, 0], pos[:, 1], pos[:, 2], output_path)

print(f"Trajectory visualization saved to {output_path}")
f.close()
import h5py,os

def read_hdf5(task):
    file_path = "/home/caohaidong/code/diffusion_policy/data/robomimic/datasets/"+task+"/ph/image_abs.hdf5"
    output = "visualize/dataset/robomimic/"+task+"_image_abs.txt"
    file_info = "file:"+task+file_path

    with open(output, 'w') as f:
        def print_hdf5_structure(root, child, print_value = False):
            # if root == '/data/demo_1' or root == '/mask': 
            #     print_value = True
            if isinstance(child, h5py.Group):
                f.write(f"{root} - Group\n")
                for key in child.keys():
                    print_hdf5_structure(f"{root}/{key}", child[key],print_value)
            elif isinstance(child, h5py.Dataset):
                f.write(f"{root} - Dataset, Size: {child.shape}, Data type: {child.dtype}\n")
                # if print_value:
                #     f.write(f"{root} - value(index 0): {child[:]}\n")
                if root[:12] == '/data/demo_0':
                    f.write(f"{root} - value(index 0~10): {child[:][0:10]}\n")
                if root[:5] == '/mask':
                    f.write(f"{root} - value: {child[:]}\n")

        with h5py.File(file_path, 'r') as file:
            f.write(file_info+"\n\n")
            f.write("HDF5 structure:\n")
            print_hdf5_structure('', file, print_value=False)

    dir, _ = os.path.split(file_path)
    command = "du -sh " + dir
    os.system(command)

if __name__ == "__main__":
    # task: square, can, lift
    task = ["square", "can", "lift"]
    # task = ["square"]
    for t in task:
        read_hdf5(t)

# content = h5py.File(file_path, 'r')
# print(content.keys())
# print(content['data'].shape)
# print(content['mask'].shape)

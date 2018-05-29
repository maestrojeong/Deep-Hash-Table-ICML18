import shutil
import os

def remove_file(path):
    print("Remove {}".format(path))
    os.remove(path)

def remove_dir(path):
    print("Remove {}".format(path))
    shutil.rmtree(path)

def copy_file(src_path, dst_path):
    print("Copy from {} to {}".format(src_path, dst_path))
    shutil.copy(src_path, dst_path)

def copy_dir(src_path, dst_path):
    print("Copy from {} to {}".format(src_path, dst_path))
    shutil.copytree(src_path, dst_path)

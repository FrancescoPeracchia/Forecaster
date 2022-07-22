
import argparse
from os import mkdir, listdir
import wget
import tqdm
from tqdm import trange
import zipfile
import pathlib
import os
from pathlib import Path
import time
import shutil


parser = argparse.ArgumentParser(description="Download KITTI")
parser.add_argument("input_file", metavar="ROOT_DIR", type=str, help="Root directory to clip's folders ")
parser.add_argument("out_dir", metavar="OUT_DIR", type=str, help="Output directory")


def main(args):
    _ensure_dir(args.out_dir)

    zip_out_path = os.path.join(args.out_dir, "zip")
    _ensure_dir(zip_out_path)

    target_folder = os.path.join(args.out_dir, "processed")
    _ensure_dir(target_folder)

    cwd = os.getcwd()

    with open(args.input_file) as f:
        lines = f.readlines()
        dim = len(lines)
        print(dim)
        print(type(dim))

        for i in trange(dim):
            drive, path_and_file = os.path.splitdrive(lines[i])
            path_, file_ = os.path.split(path_and_file)
            path_zip = os.path.join(cwd,file_)
            name_folder = file_.replace(".zip","")
            name_folder_to_delete = os.path.join(zip_out_path,'2011_09_26',name_folder)
            new_name = os.path.join(target_folder,name_folder)



    
            response = wget.download(lines[i])
            

            
            with zipfile.ZipFile(file_[:-1], 'r') as zip_ref:
                print('\n')
                zip_ref.extractall(zip_out_path)
                os.remove(path_zip[:-1])
            
            to_keep = ['image_03','oxts']
            delete_folder(name_folder_to_delete[:-1],to_keep)


            move_rename(name_folder_to_delete[:-1],new_name[:-1])

    shutil.rmtree(zip_out_path)


def _ensure_dir(dir_path):
    try:
        mkdir(dir_path)
    except FileExistsError:
        pass


def delete_folder(path,keep):
    list_=get_immediate_subdirectories(path)

    for subfolder in list_:
        if subfolder in keep:
            pass
        else:
            path_remove = os.path.join(path,subfolder)
            shutil.rmtree(path_remove)
            
def move_rename(original,target):
    shutil.move(original, target)



def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]



if __name__ == "__main__":
    main(parser.parse_args())

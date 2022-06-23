import argparse
from ctypes import POINTER
import json
import os
from multiprocessing import Pool, Value, Lock
from os import path, mkdir, listdir
import shutil
from time import process_time_ns
parser = argparse.ArgumentParser(description="Convert KITTI to coco format")
parser.add_argument("root_dir", metavar="ROOT_DIR", type=str, help="Root directory to clip's folders ")
parser.add_argument("out_dir", metavar="OUT_DIR", type=str, help="Output directory")
import cv2
import json


def main(args):
    print("Loading KITTI RAW from", args.root_dir)


    _ensure_dir(args.out_dir)
    training_path = path.join(args.out_dir, "training")
    _ensure_dir(training_path)
    validation_path = path.join(args.out_dir, "validation")
    _ensure_dir(validation_path)
    test_path = path.join(args.out_dir, "test")
    _ensure_dir(test_path)

    


    with open('data/kitti_raw/train.json', 'w') as f:
        print("The train file is created")

        training_root = path.join(args.root_dir, "training")

        #num_folders = len(get_immediate_subdirectories(args.root_dir))
        names =[]   
        id_list = [] 
        last = [] 
        
        id = 0    
        counter = 0
        for folder in get_immediate_subdirectories(training_root):
        
            folder_path = path.join(folder,'image_03/data')
            list_path = get_immediate_files(path.join(training_root, folder_path))
            folder_path = path.join(training_root,folder_path)
            
            list_path.sort()


            list_,id,id_list = get_immediate_name_files(folder,list_path,id,id_list,target_path=training_path,original_path=folder_path)
            
            last_list =[counter for i in range(len(list_))]
            counter += 1
            

            last.extend(last_list)
            names.extend(list_)
            
            
        
        print('names',names)
        print('id_list',id_list)
        print(len(last))

        data = []
        assert len(names)==len(id_list)
        assert len(last)==len(id_list)

        for id,name,end in zip (id_list,names,last):
            print('id:',id)
            print('name:',name)

            image_data = {'id': id, 'filename': name,'end_frame':end} 
            img_info = {'img_info': image_data}
            img_prefix ={'img_prefix':training_path}
            instance = {'img_info': image_data,'img_prefix':training_path}

           
            data.append(instance)
    

        
        json.dump(data,f,indent=2)
    

    for name in os.listdir(training_path):
        print('Resizing',name)
        resize(training_path,name,(2048,1024))

           
    



 
def get_immediate_name_files(names, list_path,counter,id_list, target_path = None, original_path = None):
    list_ = []
    for end in list_path:
        list_.append(names+end)
        id_list.append(counter)

        
        #new_path = os.join(target_path,names+end)
        old_path = os.path.join(original_path,end)
        old_name = os.path.join(target_path,end)
        new_name = os.path.join(target_path,names+end)
        shutil.copy(old_path, target_path)
        os.rename(old_name, new_name)

        counter+=1


    return list_,counter,id_list
              

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir)
            if name.endswith('.png')]


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    
def _ensure_dir(dir_path):
    try:
        mkdir(dir_path)
    except FileExistsError:
        pass


def resize(path,name,dim):

    name_path = os.path.join(path,name)
    img = cv2.imread(name_path,cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(name_path,resized)
    





if __name__ == "__main__":
    main(parser.parse_args())

import argparse
from ctypes import POINTER
import json
import os
from multiprocessing import Pool, Value, Lock
from os import path, mkdir, listdir
import shutil
from time import process_time_ns
import cv2
import json
import shutil

parser = argparse.ArgumentParser(description="Convert KITTI to coco format")
parser.add_argument("root_dir", metavar="ROOT_DIR", type=str, help="Root directory to clip's folders ")
parser.add_argument("out_dir", metavar="OUT_DIR", type=str, help="Output directory")


def main(args):
    print("Loading KITTI RAW from", args.root_dir)
    


    _ensure_dir(args.out_dir)

    #previous folder are deleted and recreated
    training_path = path.join(args.out_dir, "training")
    shutil.rmtree(training_path)
    _ensure_dir(training_path)

    #previous folder are deleted and recreated
    validation_path = path.join(args.out_dir, "validation")
    shutil.rmtree(validation_path)
    _ensure_dir(validation_path)

    #previous folder are deleted and recreated
    test_path = path.join(args.out_dir, "test")
    shutil.rmtree(test_path)
    _ensure_dir(test_path)


    folder_root = path.join(args.root_dir, "pra")
    num_folders = len(get_immediate_subdirectories(folder_root))
    processed = 0
    

    with open('data/kitti_raw/train.json', 'w') as f:
        print("The train file is created")
        with open('data/kitti_raw/validation.json', 'w') as v:
            print("The validation file is created")
            with open('data/kitti_raw/test.json', 'w') as t:
                print("The test file is created")

                names =[]   
                id_list = [] 
                last = [] 
                all_list_destination_path = []
                    
                id = 0    
                counter = 0

                for folder in get_immediate_subdirectories(folder_root):
                    print('processed folder ',processed, '/',num_folders )

                    if processed < 2 :
                        destination_path = training_path
                    elif processed >= 2 and processed < 4 :
                        destination_path = validation_path
                    elif processed >= 4 :
                        destination_path = test_path

                    processed +=1
                    print('PATH',destination_path)

    
                        
                    folder_path = path.join(folder,'image_03/data')
                    list_path = get_immediate_files(path.join(folder_root, folder_path))
                    folder_path = path.join(folder_root,folder_path)
                    
                    list_path.sort()


                    list_,id,id_list = get_immediate_name_files(folder,list_path,id,id_list,target_path=destination_path,original_path=folder_path)
                    
                    last_list =[counter for i in range(len(list_))]
                    list_destination_path = [destination_path for i in range(len(list_))]
                    counter += 1
                    

                    last.extend(last_list)
                    all_list_destination_path.extend(list_destination_path)
                    names.extend(list_)
                        
                        
                    
                print('names',names)
                print('id_list',id_list)
                print('lunghezza dataset',len(last))

                data_train = []
                data_val = []
                data_test = []

                assert len(names)==len(id_list)
                assert len(last)==len(id_list)
                assert len(all_list_destination_path)==len(id_list)

                for id,name,end,path_des in zip (id_list,names,last,all_list_destination_path):
                    
                    print('id:',id)
                    print('name:',name)
                    print('end-name',end)
                    #last ids works only because first are processed all training images 
                    #then all validation
                    #and finally all test
                    

                    if path_des == training_path :
                        image_data = {'id': id, 'filename': name,'end_frame':end} 
                        img_info = {'img_info': image_data}
                        img_prefix ={'img_prefix':destination_path}
                        instance = {'img_info': image_data,'img_prefix':destination_path}
                        data_train.append(instance)
                        last_id_train = id+1

                    elif path_des == validation_path :
                        image_data = {'id': id-last_id_train, 'filename': name,'end_frame':end} 
                        img_info = {'img_info': image_data}
                        img_prefix ={'img_prefix':destination_path}
                        instance = {'img_info': image_data,'img_prefix':destination_path}
                        data_val.append(instance)
                        last_id_test = id+1

                    elif path_des == test_path :
                        image_data = {'id': id-last_id_test, 'filename': name,'end_frame':end} 
                        img_info = {'img_info': image_data}
                        img_prefix ={'img_prefix':destination_path}
                        instance = {'img_info': image_data,'img_prefix':destination_path}
                        data_test.append(instance) 

                    else :           
                        print('not right path')         
                    

                
                json.dump(data_train,f,indent=2)
                json.dump(data_val,v,indent=2)
                json.dump(data_test,t,indent=2)
    

     #for name in os.listdir(destination_path):
        #print('Resizing',name)
        #resize(trainindestination_pathg_path,name,(2048,1024))

           
    



 
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

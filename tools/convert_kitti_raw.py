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
import mmcv

parser = argparse.ArgumentParser(description="Convert KITTI to coco format")
parser.add_argument("root_dir", metavar="ROOT_DIR", type=str, help="Root directory to clip's folders ")
parser.add_argument("out_dir", metavar="OUT_DIR", type=str, help="Output directory")
parser.add_argument('--resize', type=bool, default=True, help='if images should be resized by CV2')
parser.add_argument('--percentage', nargs='+', type=float, default=(0.8,0.1,0.1), help='tuple percentage to  (train,validation,test)')

def main(args):
    mgs = 80*'-'
    print(mgs)
    print("Loading KITTI RAW from", args.root_dir)
    

    RESIZE = args.resize
    PERCENTAGE = tuple(args.percentage)

    sum = PERCENTAGE[0] + PERCENTAGE[1] + PERCENTAGE[2]
    assert sum == 1 , 'splitting rate sum it should be 1 ' 
    


    _ensure_dir(args.out_dir)

    #previous folder are deleted and recreated
    print(mgs)
    print('Delete previous training data')
    training_path = path.join(args.out_dir, "training")
    shutil.rmtree(training_path)
    _ensure_dir(training_path)

    print('Delete previous validation data')
    #previous folder are deleted and recreated
    validation_path = path.join(args.out_dir, "validation")
    shutil.rmtree(validation_path)
    _ensure_dir(validation_path)

    print('Delete previous test data')
    #previous folder are deleted and recreated
    test_path = path.join(args.out_dir, "test")
    shutil.rmtree(test_path)
    _ensure_dir(test_path)


    folder_root = path.join(args.root_dir, "processed")
    num_folders = len(get_immediate_subdirectories(folder_root))

    training_len = int(num_folders*PERCENTAGE[0]) 
    validation_len = int(num_folders*PERCENTAGE[1])
    testing_len = int(num_folders*PERCENTAGE[2])
    print(mgs)
    print('CLIP for :')
    print('Training',training_len) 
    print('Validation',validation_len) 
    print('Test',testing_len) 
    print(mgs)

    #path json files
    json_train = path.join(args.out_dir,'train.json')
    json_validation = path.join(args.out_dir,'validation.json')
    json_test = path.join(args.out_dir,'test.json')

    processed = 0
    

    with open(json_train, 'w') as f:
        print("The train file is created : ", json_train)
        with open(json_validation, 'w') as v:
            print("The validation file is created : ", json_validation)
            with open(json_test, 'w') as t:
                print("The test file is created : ",json_test)
                print(mgs)

                names =[]   
                id_list = [] 
                relative_id_list = []
                last = [] 
                all_list_destination_path = []
                    
                id = 0    
                counter = 0
                print('Processing....')
                prog_bar = mmcv.ProgressBar(num_folders)

                for folder in get_immediate_subdirectories(folder_root):
                    #print('processed folder ',processed, '/',num_folders )

                    if processed < training_len :
                        destination_path = training_path
                    elif processed >= training_len and processed < training_len+validation_len :
                        destination_path = validation_path
                    elif processed >= training_len+validation_len :
                        destination_path = test_path

                    processed +=1
                    #print('PATH',destination_path)

    
                        
                    folder_path = path.join(folder,'image_03/data')
                    list_path = get_immediate_files(path.join(folder_root, folder_path))
                    folder_path = path.join(folder_root,folder_path)
                    
                    list_path.sort()


                    list_,id,id_list = get_immediate_name_files(folder,list_path,id,id_list,target_path=destination_path,original_path=folder_path)
                    
                    last_list =[counter for i in range(len(list_))]
                    relative_id = [i for i in range(len(list_))]
                    list_destination_path = [destination_path for i in range(len(list_))]
                    counter += 1
                    target = [-4,-2,0,2,4,6]
                    list_target = get_previus_past_images_id(relative_id,target)

                    

                    last.extend(last_list)
                    relative_id_list.extend(list_target)
                    all_list_destination_path.extend(list_destination_path)
                    names.extend(list_)

                    prog_bar.update()
                        
                        
                    
                #print('names',names)
                #print('id_list',id_list)
                #print('lunghezza dataset',len(last))

                data_train = []
                data_val = []
                data_test = []

                assert len(names)==len(id_list)
                assert len(last)==len(id_list)
                assert len(all_list_destination_path)==len(id_list)
                assert len(relative_id_list)==len(id_list)

                for id,name,end,list_images,path_des in zip (id_list,names,last,relative_id_list,all_list_destination_path):
                    
                    #print('id:',id)
                    #print('name:',name)
                    #print('end-name',end)
                    #last ids works only because first are processed all training images 
                    #then all validation
                    #and finally all test

                    

                    if path_des == training_path :
                   
                        image_data = {'id': id, 'filename': name,'end_frame':end}
                        if list_images is not None :
                            temp_list = [0,0,0,0,0,0] 
                            for i,temp in enumerate(target):
                                temp_list[i] = id+temp
                            list_images = temp_list
                        instance = {'img_info': image_data,'img_prefix':path_des,'list_images':list_images}
                        data_train.append(instance)
                        last_id_train = id+1

                    elif path_des == validation_path :
                        image_data = {'id': id-last_id_train, 'filename': name,'end_frame':end} 
                        if list_images is not None :
                            temp_list = [0,0,0,0,0,0]  
                            for i,temp in enumerate(target):
                                temp_list[i] = id-last_id_train+temp 
                            list_images = temp_list
                        instance = {'img_info': image_data,'img_prefix':path_des,'list_images':list_images}
                        data_val.append(instance)
                        last_id_test = id+1

                    elif path_des == test_path :
                        image_data = {'id': id-last_id_test, 'filename': name,'end_frame':end}
                        if list_images is not None :
                            temp_list = [0,0,0,0,0,0]
                            for i,temp in enumerate(target):
                                temp_list[i] = id-last_id_test+temp 
                            list_images = temp_list
                        instance = {'img_info': image_data,'img_prefix':path_des,'list_images':list_images}
                        data_test.append(instance) 

                    else :           
                        pass         
                    

                len_data_train = len(data_train)
                len_data_val = len(data_val)
                len_data_test = len(data_test)
                json.dump(data_train,f,indent=2)
                json.dump(data_val,v,indent=2)
                json.dump(data_test,t,indent=2)
    
    if RESIZE :
        print('\nResizing Train set...\n')
        prog_bar = mmcv.ProgressBar(len_data_train)
        for name in os.listdir(training_path):
            resize(training_path,name,(2048,1024))
            prog_bar.update()
            
        
        print('\nResizing Validation set...\n')
        prog_bar = mmcv.ProgressBar(len_data_val)
        for name in os.listdir(validation_path):
            resize(validation_path,name,(2048,1024))
            prog_bar.update()

        print('\nResizing Test set...\n')
        prog_bar = mmcv.ProgressBar(len_data_test)
        for name in os.listdir(test_path):
            resize(test_path,name,(2048,1024))
            prog_bar.update()

           
    



 
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



def get_previus_past_images_id(relative_ids,target):
    inferior_limit = target[0]
    superior_limit = target[5]
    list_ = []
    
    for relative_id in relative_ids :
        first =  relative_id + inferior_limit
        last =  relative_id + superior_limit
        #print('relative',relative_id)
        #print('first',first)
        #print('last',last)
       

        if first >= 0 and last <= len (relative_ids):
            list_.append(True)
        else:
            list_.append(None)
    
    return list_
        





    





if __name__ == "__main__":
    main(parser.parse_args())

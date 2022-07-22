import json
import io
import os
import argparse


def parse_args():

    parser = argparse.ArgumentParser(description="Convert KITTI to odometry format in JSON")
    parser.add_argument("path", metavar="ROOT_DIR", type=str, help="Root directory to clip's folders ")
    parser.add_argument('--target', nargs='+', type=int, default=(-5,-3,0,1,3,5), help='tuple percentage to divide (train,validation,test)')

    args = parser.parse_args()


    return args





def main():

    args = parse_args()

    path = args.path  
    target = args.target

    path_format = os.path.join(path,'2011_09_26_drive_0001_sync/oxts/dataformat.txt')
    d = read_format_odometry(path_format)
    print('PATH : ',path)
    total_relative_id = 0
    
    data_list_ = []
    list_id_relative_ = []
    list_id_total_ = []
    list_target_ = []


    for i in sorted(os.listdir(path)):

        print('folder', i)
         


        data_path = os.path.join(path,i,'oxts/data')

        list_list,list_id_relative,list_id_total,list_target,total_relative_id,keys = read_data_odometry_clip(d,data_path,total_relative_id,target)
        data_list_.extend(list_list)
        list_id_relative_.extend(list_id_relative)
        list_id_total_.extend(list_id_total)
        list_target_.extend(list_target)
 

   


    with io.open('data_odometry.json', 'w', encoding='utf-8') as w:
        
        element = {}
               
        data = {}
        


        for i in range(len(list_id_relative_)):
            
            element = {}
                    
            element['clip_relative_data'] = list_id_relative_[i]
              
            element['list'] = list_target_[i]
                  
            data[list_id_total_[i]] = element

            

            element['data'] = get_info(data_list_,i,keys)

            




        json.dump(data, w, indent=4, ensure_ascii=False)
            

def read_format_odometry(format):
    dict ={}

    with io.open('format.json', 'w', encoding='utf-8') as w:
        with open(format) as f:



            lines = f.readlines()
            for l in lines:
                split = l.split(':')

                dict[str(split[0])] = split[1]

        
        

        json.dump(dict, w, ensure_ascii=False)

        return dict


def read_data_odometry_clip(format,path,total_relative_id,target):
   
    data =[]
    counter = 0
    for i in os.listdir(path):
        counter+=1




    len_max = counter-target[5]
    len_min = -target[0]

 
    ks =format.keys()
    

    


    clip_relative_id = 0
    
    list_list_ = []
    list_id_relative = []
    list_id_total = []
    list_target = []


    
    for i in sorted(os.listdir(path)):
        

        path_text_file = os.path.join(path,i)
        with open(path_text_file) as f:
            lines = f.readlines()
            #datas only in the first line           
            split = lines[0].split(' ')


            data =[]
            for i,k in enumerate(ks):

                data.append([])
   
            
            for i,k in enumerate(ks):
                
                data[i].append(split[i])


            if clip_relative_id >= len_min and clip_relative_id < len_max:

                #print('OK',clip_relative_id)
                l = [i+total_relative_id for i in target]
                

                
            else:

                    #print('NO',clip_relative_id)
                    l = [0]


            list_id_relative.append(clip_relative_id)
            clip_relative_id += 1

            list_id_total.append(total_relative_id)
            total_relative_id += 1


            list_target.append(l)
            list_list_.append(data)


    return list_list_,list_id_relative,list_id_total,list_target,total_relative_id,ks
                

def get_info(data,i,keys):

    current_frame_data = data[i]
    data_element = {}


    for j,key in enumerate(keys):
        
        
               
        data_element[str(key)] =  current_frame_data[j]

    
   

    return data_element

        

if __name__ == "__main__":
    main()

#read_data_odometry_dataset('/home/fperacch/Forecaster/data/RAW/2011_09_26/download/zip/2011_09_26',target)


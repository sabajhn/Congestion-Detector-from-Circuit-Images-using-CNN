import os
import torch
import numpy as np
import time
from preprocess.features import matrix_coord, get_gdata, get_features1
from model.cnn_time import CNN_croute
file_address_vtr="Path"

file_address_model="./"
        
while True:
    time.sleep(0.00001)
    if(os.path.isfile(file_address_vtr + "/features/check.txt") == True and os.path.isfile(file_address_vtr + "/out.txt")== False):

        start = time.time()
        gf1=[]
        a,b,f= get_features1(get_gdata(file_address_vtr + "/features/graph_features.txt"))

        a.pop(0)
        a.pop(0)
        gf1.append(a)

        input_cnn=matrix_coord(file_address_vtr +"/features/coord.txt",f, file_address_vtr + "/features/nodef-fin.txt",file_address_vtr + "/features/nodef-fout.txt",file_address_vtr + "/features/cong_tagh.txt")

        with open(file_address_model + "gf5.txt",'r') as data_file:
            res =[line.strip().split(',') for line in data_file][0]
        res = np.asarray([float(x) for x in res] ,dtype=float)
        gf1=np.divide(gf1,res)
        gf1 = torch.tensor(gf1,dtype=float)
        with open(file_address_model + "inp5.txt" ,'r') as data_file:
            res1 =[line.strip().split(',') for line in data_file][0]

        res1 = np.asarray([float(x) for x in res1] ,dtype=float)
        # input_cnn = input_cnn[:,:,:]/res1[:]
        input_cnn=[input_cnn[i]/float(res1[i]) for i in range(len(res1))]
        input_cnn = np.reshape(input_cnn,(1,11, 81, 60)) 
        input_cnn = torch.tensor(input_cnn)
        

        model=CNN_croute() 
        model.load_state_dict(torch.load(file_address_model + "/blif_time7.pt"))
        model.eval()

        output = model(input_cnn,gf1)
        model_end = time.time()

        start_f = time.time()
        f = open(file_address_vtr + "/out.txt", "w")
        f.write(str(float(output)))
        f.close()
        end_f = time.time()



    if(os.path.isfile(file_address_vtr + "/out1.txt") == False and os.path.isfile(file_address_vtr + "/features/check.txt") == True and os.path.isfile(file_address_vtr+"/out.txt")== True):
        f1 = open(file_address_vtr + "/out1.txt", "w")
        f1.write("1")
        f1.close()
        #os.remove(file_address_vtr)


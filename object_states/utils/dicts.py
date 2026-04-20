# import required module
import sys
import os
import pandas as pd
# assign directory
directory = 'utils/annotations'

video_list = []

 
# iterate over files in
# that directory
i = 0
c = 0
cat_dicts = {}
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # print(f)
    
    if os.path.isdir(f):
        z =os.path.basename(f).split('/')[0]
        #print(z)
        
        cat_dicts[z] = []
        for r in sorted(os.listdir(f)):
            df = pd.read_csv(os.path.join(directory, filename, r), header=None, index_col=0)
            rr = os.path.basename(os.path.join(directory, filename, r)).split('.')[0]
            #print(rr)
            video_list.append(rr)
            d = df.to_dict()
            e = (rr, d[1])
            cat_dicts[z].append(e)
            ff = os.path.join(directory, r)
    i+=1


    
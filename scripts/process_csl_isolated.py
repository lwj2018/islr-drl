import os
import os.path as osp
import numpy
num_class = 500
n_person = 36

skeleton_root = "/home/liweijie/skeletons_dataset"
trainvaltest_csv = open('../csv/trainvaltest.csv','w')
trainvaltest_csv.write('foldername,label\n')
trainval_csv = open('../csv/trainval.csv','w')
trainval_csv.write('foldername,label\n')
test_csv = open('../csv/test.csv','w')
test_csv.write('foldername,label\n')
for i in range(num_class):
    c_folder = '%06d'%i
    c_path = osp.join(skeleton_root,c_folder)
    skeleton_list = os.listdir(c_path)
    skeleton_list.sort()
    for skeleton in skeleton_list:
        person = int(skeleton.split('_')[0][1:])
        skeleton_path = osp.join(c_folder,skeleton)
        record = skeleton_path + ',' + str(i) + '\n'
        if person<=n_person:
            trainval_csv.write(record)
        else:
            test_csv.write(record)
        trainvaltest_csv.write(record)
 


#%% 
#  make label for palce365
import os
from typing_extensions import assert_type 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../datasets/places365_standard')

def read_txt(path):
    f = open(path)
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    f.close()
    return lines

def write_txt(path, lines):
    f = open(path, 'w')
    f.writelines(lines)
    f.close()
    
def make_label_place365(src_path, tar_path):
    lines = read_txt(src_path)
    cate_list = [line.split('/')[1] for line in lines]
    cate_list = list(set(cate_list))
    assert(len(cate_list) == 365)
    cate_dict = {cate:i for i,cate in enumerate(cate_list)}
    new_lines = [line + '   ' + str( cate_dict[line.split('/')[1]]) + '\n'  for line in lines]
    write_txt(tar_path, new_lines)
    print(cate_list)
    

make_label_place365('train.txt', 'train_lbl.txt')
make_label_place365('val.txt', 'val_lbl.txt')


#%%
# make label for anp
import os 
train_root = "/home/ubuntu/ftl/dataset/senti/anp/train"
test_root = "/home/ubuntu/ftl/dataset/senti/anp/test"
train_cate_names = os.listdir(train_root)
test_cate_names = os.listdir(test_root)
assert(set(train_cate_names) == set(test_cate_names))

cate_names = train_cate_names
cate_names = sorted(cate_names)
# print(cate_names)
cate_label_dict = {cate:label for label, cate in enumerate(cate_names)}
train_lines = []
test_lines = []

for cate, label, in cate_label_dict.items():
    cate_folder_path = os.path.join(train_root, cate)
    img_list = os.listdir(cate_folder_path)
    for img_name in img_list:
        line = os.path.join('./train', cate, img_name)
        line = line + '   ' + str(label) + '\n'
        train_lines.append(line)

for cate, label, in cate_label_dict.items():
    cate_folder_path = os.path.join(test_root, cate)
    img_list = os.listdir(cate_folder_path)
    for img_name in img_list:
        line = os.path.join('./test', cate, img_name)
        line = line + '   ' + str(label) + '\n'
        test_lines.append(line)

f = open("/home/ubuntu/ftl/dataset/senti/anp/train.txt",'w')
f.writelines(train_lines)
f.close()

f = open("/home/ubuntu/ftl/dataset/senti/anp/test.txt",'w')
f.writelines(test_lines)
f.close()

print()
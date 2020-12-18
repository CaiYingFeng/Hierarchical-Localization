import h5py
import numpy as np
from pathlib import Path
#打开文件
f = h5py.File('/media/autolab/disk_3T/caiyingfeng/localization/out/exports/netvlad/query0808_db.h5','r')
db_prefix='query'
names = []
f.visititems(
    lambda _, obj: names.append(obj.parent.name.strip('/'))
    if isinstance(obj, h5py.Dataset) else None)
names = list(set(names))
db_names = [n for n in names if n.startswith(db_prefix)]
base_dir='/media/autolab/disk_3T/caiyingfeng/localization/out/exports/netvlad/aachen/new_netvlad/query_0808'
k=0
for i in db_names:
    k=k+1
    des=f[i]['global_descriptor'].__array__()
    mypre={'global_descriptor':des}
    mypre['input_shape']=[600,960,1]
    name=i.split('/',-1)[-1][:-4]

    Path(base_dir, Path(name).parent).mkdir(parents=True, exist_ok=True)
    np.savez(Path(base_dir, '{}.npz'.format(name)), **mypre) 

    if k % 50 == 0 :
        print("==> Batch ({}/{})".format(k,len(db_names)), flush=True)
        print(i)
    
    
# # print(f.keys())
# # print(f['db'][:])
# # 遍历文件中的一级组
# for group in f.keys():
#     print (group)
#     #根据一级组名获得其下面的组
#     group_read = f[group]
#     #遍历该一级组下面的子组
#     for subgroup in group_read.keys():
#         print (subgroup)    
#         #根据一级组和二级组名获取其下面的dataset          
#         dset_read = f[group+'/'+subgroup]                           
#         #遍历该子组下所有的dataset
#         for dset in dset_read.keys():
#             #获取dataset数据
#             dset1 = f[group+'/'+subgroup+'/'+dset]
#             print (dset1.name)
#             data = np.array(dset1)
#             print (data.shape)
#             x = data[...,0]
#             y = data[...,1]        

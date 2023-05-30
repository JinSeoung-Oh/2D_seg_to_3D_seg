import json

def write_ply(verts, colors, indices, out_path):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    print('.....')
    file = open(out_path, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
   
    for vert, color in zip(verts, colors): #<-- check this.. it is not matched with index
        file.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(vert[0], vert[1], vert[2], int(color[0]), int(color[1]), int(color[2])))
                                                           
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))

    file.close()
                   
    print('write_ply is done')

with open('./Grounded-Segment-Anything/outputs/mask_car.json', 'r') as f:
    data = json.load(f)
   
data = data[1:]
color_map = np.zeros_like(pcds)

for k in range(0,465750-1,1):  # image h * w (1242*375 = 465750)
    color_map[k] = [153,255,153]

# first obejct 
indexs = []
max_ = 1242
for coor in data[1]['mask_coor']:
    y = coor[1]
    x = coor[0]
    if x != 0:
        index = max_*(y-1)+x
    else:
        index = coor[1]
    indexs.append(index)
   
   
color = [0,255,0]
for index in indexs:
    color_map[index] = color

# scond objet 
indexs = []
max_ = 1242
for coor in data[2]['mask_coor']:
    y = coor[1]
    x = coor[0]
    if x != 0:
        index = max_*(y-1)+x
    else:
        index = coor[1]
    indexs.append(index)
   
   
color = [0,0,255]
for index in indexs:
    color_map[index] = color


write_ply(ptcd, color_map, None, './matching_all.ply')
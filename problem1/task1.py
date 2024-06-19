from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


data = load_data('data/data.p') # Change to data.p for your final submission 

points = data['velodyne']

image = data['image_2']

K = data['K_cam2']

T_cam2_velo = data['T_cam2_velo']
T_cam0_velo = data['T_cam0_velo']

P = data['P_rect_20']

markers = []
colors = []

#---question 1

for i in range(len(points)):
    #check that point.x > 0 
    # if point.x <= 0 then it is behind the camera ==> hidden 
    if points[i][0] > 0:   
        point = np.transpose(np.append(points[i][0:3], 1))
        point_cam2_ref = T_cam2_velo @ point
        #obtain the point in the camera 2 frame ref
        point_cam2_plane = K @ point_cam2_ref[0:3]
        #project to the camera plane
        point_norm = point_cam2_plane[0:2] / point_cam2_plane[2]
        #normalize pixel coord
        markers.append(point_norm)

        label = int(data['sem_label'][i])
        color_map = data['color_map'][label]
        color = [color_map[2], color_map[1], color_map[0]]

        colors.append(color)

markers = np.array(markers)
colors = np.array(colors) / 255
#divide with 255 to make sure color value will be correct

plt.scatter(markers[:, 0], markers[:, 1], s=0.07, c=colors, marker='.')

#---question 2

objects = data['objects']
bounding_box = []
i = -1

for car in objects:
    i += 1

    height = car[8]
    length = car[9]
    width = car[10]
    angle = car[14]

    bounding_box_vertices = [[-width/2, 0, length/2], 
                [width/2, 0, length/2], 
                [width/2, 0, -length/2], 
                [-width/2, 0, -length/2],
                [-width/2, -height, length/2], 
                [width/2, -height, length/2], 
                [width/2, -height, -length/2], 
                [-width/2, -height, -length/2]]

    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    #rotation angle around y axis
    transf = np.array([car[11], car[12], car[13]]) 
    transl = np.insert(rotation, 3, transf, axis=1)
    #translation using center coordinates
    T_cam0_car = np.insert(transl, 3, [0, 0, 0, 1], axis=0)
    #transformation matrix from camera 0 to car

    T_cam2_cam0 = T_cam2_velo @ np.linalg.inv(T_cam0_velo) 
    #transformation matrix from camera 0 to camera 2
    T_cam2_car = T_cam2_cam0 @ T_cam0_car #T2c = T20*T0c
    #transformation matrix from car to camera 2

    for b in bounding_box_vertices:
        bounding_box_point = np.append(b, 1)
        bounding_box_point_cam2_ref = T_cam2_car @ bounding_box_point
        #get point in the camera 2 frame ref
        bounding_box_point_cam2_plane = K @ bounding_box_point_cam2_ref[0:3]
        #project to camera plane
        bounding_box_point_norm = bounding_box_point_cam2_plane[0:2] / bounding_box_point_cam2_plane[2]
        #normalize coordinates

        bounding_box.append(bounding_box_point_norm)

    plt.gca().add_patch(Polygon(bounding_box[i*8:i*8+4], closed=True, fill=False, ec='lawngreen', lw=1.8))
    plt.gca().add_patch(Polygon(bounding_box[i*8+4:i*8+8], closed=True, fill=False, ec='lawngreen', lw=1.8))

    plt.gca().add_patch(Polygon([bounding_box[i*8], bounding_box[i*8+4], bounding_box[i*8+7], bounding_box[i*8+3]], closed=True, fill=False, ec='lawngreen', lw=1.8))
    plt.gca().add_patch(Polygon([bounding_box[i*8+1], bounding_box[i*8+5], bounding_box[i*8+6], bounding_box[i*8+2]], closed=True, fill=False, ec='lawngreen', lw=1.8))

bounding_box = np.array(bounding_box)

plt.imshow(image)
plt.show()


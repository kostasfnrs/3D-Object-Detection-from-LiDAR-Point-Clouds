# Computer Vision and Artificial Intelligence for Autonomous Cars
# Material for Problem 1 of Project 2
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import os
from load_data import load_data

class Visualizer():
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                        parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.sem_vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.sem_vis)
        visuals.XYZAxis(parent=self.view.scene)
        
        # Object Detection Visualizer
        self.obj_vis = visuals.Line()
        self.view.add(self.obj_vis)
        self.connect = np.asarray([[6, 5], [6, 7], [6, 2],
                                   [4, 5], [4, 7], [4, 0],
                                   [1, 5], [1, 2], [1, 0],
                                   [3, 7], [3, 2], [3, 0]])

    def update(self, points, sem_label, color_map):
        '''
        :param points: point cloud data
                        shape (N, 3)          
        Task: Change this function such that each point
        is colored depending on its semantic label
        '''
        colors = np.zeros([len(points), 3])
        for i in range(len(points)):
            label = sem_label[i]
            color = color_map[int(label)]
            color = [color[2], color[1], color[0]]
            colors[i][:] = np.array(color) / 255

        self.sem_vis.set_data(points, size=3, face_color=colors)
    
    def update_boxes(self, corners):
        '''
        :param corners: corners of the bounding boxes
                        shape (N, 8, 3) for N boxes
        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        If you plan to use a different order, you can
        change self.connect accordinly.
        '''
        for i in range(corners.shape[0]):
            connect = np.concatenate((connect, self.connect+8*i), axis=0) \
                      if i>0 else self.connect
        self.obj_vis.set_data(corners.reshape(-1,3),
                              connect=connect,
                              width=2,
                              color=[0,1,0,1])

if __name__ == '__main__':
    data = load_data('data/data.p') # Change to data.p for your final submission 
    
    visualizer = Visualizer()

    sem_label = data['sem_label']
    color_map = data['color_map']
    objects = data['objects']
    T_cam2_velo = data['T_cam2_velo']
    T_cam0_velo = data['T_cam0_velo']
    visualizer.update(data['velodyne'][:,:3], sem_label, color_map)

    bounding_box = np.zeros((len(objects), 8, 3))
    i = -1
    for car in objects:
        i += 1
        height = car[8]
        length = car[9]
        width = car[10]
        angle = car[14]

        bounding_box_vertices = [[-width / 2, 0, length / 2],
                                [width / 2, 0, length / 2], 
                                [width / 2, 0, -length / 2],
                                [-width / 2, 0, -length / 2],
                                [-width / 2, -height, length / 2],
                                [width / 2, -height, length / 2], 
                                [width / 2, -height, -length / 2],
                                [-width / 2, -height, -length / 2]]

        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        #rotation angle around y axis
        transf = np.array([car[11], car[12], car[13]])
        transl = np.insert(rotation, 3, transf, axis=1)
        #translation using center coordinates
        T_cam0_car = np.insert(transl, 3, [0, 0, 0, 1],axis=0)
        #transformation matrix from camera 0 to car
        T_velo_car = np.linalg.inv(T_cam0_velo) @ T_cam0_car
        #transformation matrix from car to velodyne

        bounding_box_points = np.insert(bounding_box_vertices, 3, 1, axis=1)
        bounding_box_points_velo = T_velo_car @ np.transpose(bounding_box_points)
        #get point in the velodyne frame ref
        bounding_box_points_velo = np.transpose(bounding_box_points_velo[0:3, :])

        bounding_box[i][:][:] = bounding_box_points_velo

    visualizer.update_boxes(bounding_box)
    '''
    Task: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    '''
    vispy.app.run()





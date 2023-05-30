import numpy as np
import open3d as o3d

matrix = o3d.cuda.pybind.camera.PinholeCameraIntrinsic(width = 1242, height =  375, fx = 721.5337, fy = 721.5377, cx = 609.5593, cy = 172.854)

color_raw = o3d.io.read_image("./Grounded-Segment-Anything/assets/road.png")
depth_raw = o3d.io.read_image('./test_image_road.png')
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        matrix))
    # Flip it, otherwise the pointcloud will be upside down
print(pcd)
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
points3d = pcd.points
points3d = np.asarray(points3d)
points = np.reshape(points3d, (-1,3))

pcds = []
for point in points:
    pcds.append(point)
   
point_cloud_data = o3d.geometry.PointCloud()
point_cloud_data.points = o3d.utility.Vector3dVector(pcds)
o3d.io.write_point_cloud("./result_1.ply", point_cloud_data)
o3d.visualization.draw_geometries([point_cloud_data])

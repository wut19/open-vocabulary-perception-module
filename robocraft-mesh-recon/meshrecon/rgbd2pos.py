import open3d as o3d
import os, sys
from datetime import datetime
import numpy as np
import pymeshfix
import pyvista as pv
from pysdf import SDF
from typing import Dict

visualize = False
o3d_write = False

n_points = 6000

########################### visualization funcs ##########################

def o3d_visualize(display_list):
    if o3d_write:
        cd = os.path.dirname(os.path.realpath(sys.argv[0]))
        time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
        image_path = os.path.join(cd, '..', '..', 'images', f'{time_now}.png')

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for geo in display_list:
            vis.add_geometry(geo)
            vis.update_geometry(geo)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(image_path)
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries(display_list, mesh_show_back_face=True)


def visualize_points(ax, all_points, n_points):
    points = ax.scatter(all_points[:n_points, 0], all_points[:n_points, 2], all_points[:n_points, 1], c='b', s=10)
    shapes = ax.scatter(all_points[n_points:, 0], all_points[n_points:, 2], all_points[n_points:, 1], c='r', s=10)
    
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.25  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    ax.invert_yaxis()

    return points, shapes
################################## end #################################

########################### rgbd to point cloud ##########################

def gen_3D_za(intrinsic, extrinsic, rgb, depth, w=512, h=512):
    fx = fy = intrinsic[0, 0]
    cx = cy = intrinsic[0, 2]

    cam = o3d.camera.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], fx, fy, cx, cy)
    # extrinsic = get_ext(camera_rot, np.array(camera_pos))
    RGB = o3d.geometry.Image(np.ascontiguousarray(np.rot90(rgb,0,(0,1))).astype(np.uint8))
    DEPTH = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(RGB, DEPTH, depth_scale=1., depth_trunc=np.inf, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)

    # pcd.transform(np.linalg.inv(extrinsic).T)
    pcd.transform(extrinsic)

    return pcd

def im2threed(rgb, depth, mask, cam_params):
    n_cam = rgb.shape[0]
    pcd_all = o3d.geometry.PointCloud()
    for i in range(n_cam):
        pcd = gen_3D_za(intrinsic=cam_params['intrinsic'], extrinsic=cam_params[f'cam{i+1}_ext'], 
                        rgb=rgb[i], depth=depth[i])
        points_scene = np.ascontiguousarray(pcd.points)
        points_masked = points_scene[mask[i].reshape(-1,)]
        colors_masked = rgb[i][mask[i]].reshape(-1,3) / 255

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_masked)
        pcd.colors = o3d.utility.Vector3dVector(colors_masked)
        pcd_all += pcd

    return pcd_all

################################## end #################################

########################### mesh reconstruction ##########################

def flip_inward_normals(pcd, center, threshold=0.7):
    # Flip normal if normal points inwards by changing vertex order
    # https://math.stackexchange.com/questions/3114932/determine-direction-of-normal-vector-of-convex-polyhedron-in-3d
    
    # Get vertices and triangles from the mesh
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    # For each triangle in the mesh
    flipped_count = 0
    for i, n in enumerate(normals):
        # Compute vector from 1st vertex of triangle to center
        norm_ref = points[i] - center
        # Compare normal to the vector
        if np.dot(norm_ref, n) < 0:
            # Change vertex order to flip normal direction
            flipped_count += 1 
            if flipped_count > threshold * normals.shape[0]:
                normals = np.negative(normals)
                break

    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def mesh_reconstruct(pcd, algo="filter", alpha=0.5, depth=8, visualize=False):
    if algo == "filter":
        point_cloud = pv.PolyData(np.asarray(pcd.points))
        surf = point_cloud.reconstruct_surface()

        mf = pymeshfix.MeshFix(surf)
        mf.repair(verbose=True,remove_smallest_components=True)
        pymesh = mf.mesh

        if visualize:
            pl = pv.Plotter()
            pl.add_mesh(point_cloud, color='k', point_size=10)
            pl.add_mesh(pymesh)
            pl.add_title('Reconstructed Surface')
            pl.show()

        mesh = pymesh
    else:
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        center = pcd.get_center()
        pcd = flip_inward_normals(pcd, center)

        if algo == "ball_pivot":
            radii = [0.005, 0.01, 0.02, 0.04, 0.08]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
        elif algo == "alpha_shape":
            tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha, tetra_mesh, pt_map)
        elif algo == "poisson":
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        else:
            raise NotImplementedError

        mesh.paint_uniform_color([0,1,0])
        mesh.compute_vertex_normals()

        if visualize:
            o3d_visualize([pcd, mesh])
    
    return mesh

################################## end #################################

########################### resample and position estimation ##########################

def pos_estimate(mesh):
    
    pos = mesh.get_center()
    return pos

################################## end #################################

########################### rgbd to postition ##########################

def rgbd2pos(rgb, depth, mask, cam_params):
    
    pcd = im2threed(rgb, depth, mask, cam_params)
    _,index = pcd.remove_statistical_outlier(nb_neighbors = 50, std_ratio= 1.0)
    pcd = pcd.select_by_index(index)
    mesh = mesh_reconstruct(pcd)
    pos = pos_estimate(mesh)
    return pos



def estimate_pos(obs, cam_params):
    return rgbd2pos(obs['rgb'], obs['depth'], obs['mask'], cam_params)


if __name__ == "__main__":
    # rgb = np.load('rgb.npy')
    # depth = np.load('depth.npy')
    # mask = np.load('mask.npy')
    # rgb1 = np.load('rgb1.npy')
    # depth1 = np.load('depth1.npy')
    # mask1 = np.load('mask1.npy')

    cam_params = {}
    cam_params['cam1_ext'] = np.array([[ 1.19209290e-07, -4.22617942e-01, -9.06307936e-01,  1.34999919e+00],
                              [-1.00000000e+00, -5.96046448e-07,  1.49011612e-07,  3.71546562e-08],
                              [-5.66244125e-07,  9.06307936e-01, -4.22617912e-01,  1.57999933e+00],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    cam_params['cam2_ext'] = np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.00000012e-01],
                                        [ 0.00000000e+00, -1.19209290e-07, -1.00000012e+00,  1.60000002e+00],
                                        [ 0.00000000e+00,  1.00000012e+00, -1.19209290e-07,  1.20000005e+00],
                                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    cam_params['intrinsic'] = np.array([[-703.3542416,    0.      ,   256.       ],
                               [   0.   ,     -703.3542416 , 256.       ],
                               [   0.    ,       0.     ,      1.       ]])
    
    np.save('cam_params.npy',cam_params)
    # rgb = np.stack([rgb,rgb1],axis=0)
    # depth = np.stack([depth,depth1],axis=0)
    # mask = np.stack([mask,mask1],axis=0)

    # pcd = im2threed(rgb,depth,mask,cam_params)
    
    # _,index = pcd.remove_statistical_outlier(nb_neighbors = 50, std_ratio= 1.0)
    # pcd = pcd.select_by_index(index)
    # # o3d_visualize([pcd])
    # mesh = mesh_reconstruct(pcd,algo='alpha_shape',alpha=0.5)
    # # o3d_visualize([mesh])
    # # mash = mesh.filter_smooth_simple(10)
    # # o3d_visualize([mesh])
    # pos = pos_estimate(mesh)
    # print(pos)
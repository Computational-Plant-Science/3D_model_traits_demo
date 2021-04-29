"""
Version: 1.0

Summary: Build graph from STL file and visualize mesh and nodes in 3D

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

python3 section_scan.py -p ~/example/ -m test.stl


argument:
("-p", "--path", required=True,    help="path to *.stl model file")
("-m", "--model", required=True,    help="file name")

"""

def stl_graph_vis(mesh_file):

    mesh = trimesh.load_mesh(mesh_file)
 
    # VISUALIZE RESULT
    # make the sphere transparent-ish
    mesh.visual.face_colors = [100, 100, 100, 100]
    # Path3D with the path between the points
    
    #path_visual = trimesh.load_path(mesh.vertices[path])
    
    #vertices_visual = trimesh.load_path(mesh.vertices)
    vertices_visual = []
    
    # visualizable two points
    #points_visual = trimesh.points.PointCloud(mesh.vertices[[start, end]])
    points_visual = trimesh.points.PointCloud(mesh.vertices)

    # create a scene with the mesh, path, and points
    scene = trimesh.Scene([
        points_visual,
        vertices_visual,
        mesh])

    scene.show(smooth=False)

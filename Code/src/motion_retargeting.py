import numpy as np
from numba import njit,prange

def rotation_matrix_from_vectors(a, b):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """

    if np.array_equal(a , b) or np.array_equal(a , np.zeros(3)) or np.array_equal(b , np.zeros(3)):
        return np.eye(3)
    a = a/np.linalg.norm(a,keepdims=True)
    b = b/np.linalg.norm(b,keepdims=True)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v,ord=2)

    if s == 0:
        return np.eye(3)
    else:
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

@njit(parallel=True)
def add_transformation(new_points,points,rotation_matrix,old_position,new_position,attachment):

    N = points.shape[0]

    for n in prange(N):
        per_bone_position = rotation_matrix.dot(points[n,:]-old_position) + new_position 
        # print(per_bone_position)
        new_points[n,:] += attachment[n]*per_bone_position

    return new_points

def update_vertices(points,parent_array,old_joints,new_joints,attachment):
    """
        Perform linear blend skinning using old joints to convert to get new joints
    """

    J = len(parent_array)
    # N = points.shape[0]

    new_skeleton = np.zeros_like(old_joints)
    new_skeleton[0,:] = new_joints[0,:]
    new_points = np.zeros_like(points)

    for j in range(1,J):
        p = parent_array[j]
        rotation_matrix = rotation_matrix_from_vectors(old_joints[j,:]-old_joints[p,:],new_joints[j,:]-new_joints[p,:])
        
        new_skeleton[j,:] = rotation_matrix.dot(old_joints[j,:]-old_joints[p,:]) + new_skeleton[p,:]
        new_points = add_transformation(new_points,points,rotation_matrix,old_joints[p,:],new_skeleton[p,:],attachment[:,j-1])


    return new_points,new_skeleton

def get_mesh_motion(points,parent_array,old_joints,skeleton_motion,attachment):
    """
        Given the skeleton motion get the position of the mesh vertices 
    """
    mesh_motion = [] 
    target_skeleton_motion = []
    
    # Loop over all timesteps to get the new position of each mesh vertice
    for new_joints in skeleton_motion:
        # print("???")
        new_points,new_skeleton = update_vertices(points,parent_array,old_joints,new_joints,attachment)

        # print(new_points.shape,new_skeleton.shape)
        mesh_motion.append(new_points)
        target_skeleton_motion.append(new_skeleton)
        # print(f"Completed:{len(mesh_motion)}/{len(skeleton_motion)}")
        
        # if len(mesh_motion) > 10:
        #     break

    return np.array(mesh_motion),np.array(target_skeleton_motion)
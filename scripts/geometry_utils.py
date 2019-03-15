import numpy as np
import os
import json

"""
    This file contains some helper functions for geometry processing
    Author: Kaichun Mo
"""

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; normals = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)
    return v, f

def load_off(fn):
    fin = open(fn, 'r')
    line = fin.readline()
    line = fin.readline()
    num_vertices = int(line.split()[0])
    num_faces = int(line.split()[1])

    vertices = np.zeros((num_vertices, 3)).astype(np.float32)
    for i in range(num_vertices):
        vertices[i, :] = np.float32(fin.readline().split())

    faces = np.zeros((num_faces, 3)).astype(np.int32)
    for i in range(num_faces):
        faces[i, :] = np.int32(fin.readline().split()[1:]) + 1

    fin.close()

    return faces, vertices

def normalize_pts(pts):
    out = np.array(pts, dtype=np.float32)
    center = np.mean(out, axis=0)
    out -= center
    scale = np.sqrt(np.max(np.sum(out**2, axis=1)))
    out /= scale
    return out

def rotate_pts(pts, theta=0, phi=0):
    rotated_data = np.zeros(pts.shape, dtype=np.float32)

    # rotate along y-z axis
    rotation_angle = phi / 90 * np.pi / 2
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, sinval],
                                [0, -sinval, cosval]])
    rotated_pts = np.dot(pts, rotation_matrix)

    # rotate along x-z axis
    rotation_angle = theta / 360 * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_pts = np.dot(rotated_pts, rotation_matrix)
    return rotated_pts

def load_pts(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines], dtype=np.float32)
        return pts

def load_pts_nor(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines], dtype=np.float32)
        nor = np.array([[float(line.split()[3]), float(line.split()[4]), float(line.split()[5])] for line in lines], dtype=np.float32)
        return pts, nor

def load_label(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        label = np.array([int(line) for line in lines], dtype=np.int32)
        return label

def export_obj(out, v, f):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

def export_label(out, label):
    with open(out, 'w') as fout:
        for i in range(label.shape[0]):
            fout.write('%d\n' % label[i])

def export_pts(out, v):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))

def export_pts_with_normal(out, v, n):
    assert v.shape[0] == n.shape[0], 'v.shape[0] != v.shape[0]'

    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], n[i, 0], n[i, 1], n[i, 2]))

def export_ply(out, v):
    with open(out, 'w') as fout:
        fout.write('ply\n');
        fout.write('format ascii 1.0\n');
        fout.write('element vertex '+str(v.shape[0])+'\n');
        fout.write('property float x\n');
        fout.write('property float y\n');
        fout.write('property float z\n');
        fout.write('end_header\n');

        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))

def export_ply_with_label(out, v, l):
    num_colors = len(colors)
    with open(out, 'w') as fout:
        fout.write('ply\n');
        fout.write('format ascii 1.0\n');
        fout.write('element vertex '+str(v.shape[0])+'\n');
        fout.write('property float x\n');
        fout.write('property float y\n');
        fout.write('property float z\n');
        fout.write('property uchar red\n');
        fout.write('property uchar green\n');
        fout.write('property uchar blue\n');
        fout.write('end_header\n');

        for i in range(v.shape[0]):
            cur_color = colors[l[i]%num_colors]
            fout.write('%f %f %f %d %d %d\n' % (v[i, 0], v[i, 1], v[i, 2], \
                    int(cur_color[0]*255), int(cur_color[1]*255), int(cur_color[2]*255)))


def export_ply_with_normal(out, v, n):
    assert v.shape[0] == n.shape[0], 'v.shape[0] != v.shape[0]'

    with open(out, 'w') as fout:
        fout.write('ply\n');
        fout.write('format ascii 1.0\n');
        fout.write('element vertex '+str(v.shape[0])+'\n');
        fout.write('property float x\n');
        fout.write('property float y\n');
        fout.write('property float z\n');
        fout.write('property float nx\n');
        fout.write('property float ny\n');
        fout.write('property float nz\n');
        fout.write('end_header\n');

        for i in range(v.shape[0]):
            fout.write('%f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], n[i, 0], n[i, 1], n[i, 2]))

def export_pts_with_color(out, pc, label):
    num_point = pc.shape[0]
    with open(out, 'w') as fout:
        for i in range(num_point):
            cur_color = label[i]
            fout.write('%f %f %f %d %d %d\n' % (pc[i, 0], pc[i, 1], pc[i, 2], cur_color[0], cur_color[1], cur_color[2]))

def export_pts_with_label(out, pc, label, base=0):
    num_point = pc.shape[0]
    num_colors = len(colors)
    with open(out, 'w') as fout:
        for i in range(num_point):
            cur_color = colors[label[i]%num_colors]
            fout.write('%f %f %f %f %f %f\n' % (pc[i, 0], pc[i, 1], pc[i, 2], cur_color[0], cur_color[1], cur_color[2]))

def export_pts_with_keypoints(out, pc, kp_list):
    num_point = pc.shape[0]
    with open(out, 'w') as fout:
        for i in range(num_point):
            if i in kp_list:
                color = [1.0, 0.0, 0.0]
            else:
                color = [0.0, 0.0, 1.0]

            fout.write('%f %f %f %f %f %f\n' % (pc[i, 0], pc[i, 1], pc[i, 2], color[0], color[1], color[2]))

def compute_normal(pts, neighbor=50):
    l = pts.shape[0]
    assert(l > neighbor)
    t = np.sum(pts**2, axis=1)
    A = np.tile(t, (l, 1))
    C = np.array(A).T
    B = np.dot(pts, pts.T)
    dist = A - 2 * B + C

    neigh_ids = dist.argsort(axis=1)[:, :neighbor]
    vec_ones = np.ones((neighbor, 1)).astype(np.float32)
    normals = np.zeros((l, 3)).astype(np.float32)
    for idx in range(l):
        D = pts[neigh_ids[idx, :], :]
        cur_normal = np.dot(np.linalg.pinv(D), vec_ones)
        cur_normal = np.squeeze(cur_normal)
        len_normal = np.sqrt(np.sum(cur_normal**2))
        normals[idx, :] = cur_normal / len_normal
        
        if np.dot(normals[idx, :], pts[idx, :]) < 0:
            normals[idx, :] = -normals[idx, :]

    return normals

def transfer_label_from_pts_to_obj(vertices, faces, pts, label):
    assert pts.shape[0] == label.shape[0], 'ERROR: #pts != #label'
    num_pts = pts.shape[0]

    num_faces = faces.shape[0]
    face_centers = []
    for i in range(num_faces):
        face_centers.append((vertices[faces[i, 0]-1, :] + vertices[faces[i, 1]-1, :] + vertices[faces[i, 2]-1, :]) / 3)
    face_center_array = np.vstack(face_centers)

    A = np.tile(np.expand_dims(np.sum(face_center_array**2, axis=1), axis=0), [num_pts, 1])
    B = np.tile(np.expand_dims(np.sum(pts**2, axis=1), axis=1), [1, num_faces])
    C = np.dot(pts, face_center_array.T)
    dist = A + B - 2*C

    lid = np.argmax(-dist, axis=0)
    face_label = label[lid]
    return face_label

def detect_connected_component(vertices, faces, face_labels=None):
    edge2facelist = dict()
    
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]
    
    bar = progressbar.ProgressBar()
    face_id_list = []
    for face_id in bar(range(num_faces)):
        f0 = faces[face_id, 0] - 1
        f1 = faces[face_id, 1] - 1
        f2 = faces[face_id, 2] - 1
        id_list = np.sort([f0, f1, f2])
        s0 = id_list[0]
        s1 = id_list[1]
        s2 = id_list[2]
        
        key1 = '%d_%d' % (s0, s1)
        if key1 in edge2facelist.keys(): edge2facelist[key1].append(face_id)
        else: edge2facelist[key1] = [face_id]

        key2 = '%d_%d' % (s1, s2)
        if key2 in edge2facelist.keys(): edge2facelist[key2].append(face_id)
        else: edge2facelist[key2] = [face_id]

        key3 = '%d_%d' % (s0, s2)
        if key3 in edge2facelist.keys(): edge2facelist[key3].append(face_id)
        else: edge2facelist[key3] = [face_id]

        face_id_list.append([key1, key2, key3])

    face_used = np.zeros((num_faces), dtype=np.bool);
    face_seg_id = np.zeros((num_faces), dtype=np.int32); cur_id = 0;

    new_part = False
    for i in range(num_faces):
        q = deque()
        q.append(i)
        while len(q) > 0:
            face_id = q.popleft()
            if not face_used[face_id]:
                face_used[face_id] = True
                new_part = True
                face_seg_id[face_id] = cur_id 
                for key in face_id_list[face_id]:
                    for new_face_id in edge2facelist[key]:
                        if not face_used[new_face_id] and (face_labels == None or \
                                face_labels[new_face_id] == face_labels[face_id]):
                            q.append(new_face_id)

        if new_part: 
            cur_id += 1
            new_part = False

    return face_seg_id

def calculate_two_pts_distance(pts1, pts2):
    A = np.tile(np.expand_dims(np.sum(pts1**2, axis=1), axis=-1), [1, pts2.shape[0]])
    B = np.tile(np.expand_dims(np.sum(pts2**2, axis=1), axis=0), [pts1.shape[0], 1])
    C = np.dot(pts1, pts2.T)
    dist = A + B - 2 * C
    return dist

def propagate_pts_seg_from_another_pts(ori_pts, ori_seg, tar_pts):
    dist = calculate_two_pts_distance(ori_pts, tar_pts)
    idx = np.argmin(dist, axis=0)
    return ori_seg[idx]

def remove_isolated_vertices(v, f):
    used = np.zeros((v.shape[0]+1), dtype=np.bool)
    for i in range(f.shape[0]):
        used[f[i, 0]] = True
        used[f[i, 1]] = True
        used[f[i, 2]] = True
    new_fids = np.cumsum(used)
    f = new_fids[f]
    v = v[used[1:], :]
    return v, f


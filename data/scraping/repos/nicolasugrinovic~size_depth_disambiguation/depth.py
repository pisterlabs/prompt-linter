import torch
from util.misc import save_points, read_json
import numpy as np
import os
from .misc import plot_joints_cv2


def get_sign_matix(gt_trans_):
    all_depts = gt_trans_[:, 2]
    z_diffs = all_depts[:, None] - all_depts[None, :]
    sign_matrix = z_diffs / abs(z_diffs + 0.000000001)
    return np.ceil(sign_matrix)

def get_sign_matix_from_depths(all_depts):
    z_diffs = all_depts[:, None] - all_depts[None, :]
    sign_matrix = z_diffs / abs(z_diffs + 0.000000001)
    return np.ceil(sign_matrix)


def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]

def upper_tri_masking_torch(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A * torch.Tensor(mask).cuda()

def mask_distance_reverse(masks_person_arr, kpts_int, w, h):
    scores = []
    for i, persons_mask in enumerate(masks_person_arr):
        this_mask = []
        for smpl_joints_as_idx in kpts_int[..., :2]:
            xs, ys = zip(*smpl_joints_as_idx)
            xs = np.clip(xs, 0, w - 1)
            ys = np.clip(ys, 0, h - 1)
            smpl_joints_as_idx = np.array([ys, xs]).T
            joint_in_mask = []
            for idx in smpl_joints_as_idx:
                mask_at_joint = persons_mask[tuple(idx)]
                joint_in_mask.append(mask_at_joint)
            joint_in_mask = np.array(joint_in_mask).sum()
            this_mask.append(joint_in_mask)
            # print(joint_in_mask)
        this_mask = np.array(this_mask)
        scores.append(this_mask)
    scores_arr = np.stack(scores, 1)
    maxsc = scores_arr.max(1)
    cost = maxsc[:, None] - scores_arr

    return cost


def vis_proj_joints_t(image, joints_projected, gt_keypoints, do_plot=True):
    '''
    Args:
        image:
        joints_projected: tensor de [B, njoints, 3]
        gt_keypoints:
    Returns:
    '''
    init_joints = joints_projected.int().cpu()
    if gt_keypoints is None:
        init_joints = init_joints
    else:
        conf = gt_keypoints[..., -1].cpu()
        vis = conf > 0.0
        init_joints = init_joints * vis[..., None]
    out_img = plot_joints_cv2(image, init_joints, do_plot, with_text=True)
    return out_img


def get_person_depth(d_img, masks_person, i):
    depth_img = torch.Tensor(d_img).cuda()
    depth_person = masks_person[i] * depth_img
    sum = depth_person.sum()
    n = masks_person[i].sum()
    calc_mean = sum / n
    return calc_mean, depth_person

# plot(depth_person.cpu())

def get_depths_from_crops(masked_depth, persons_mask, bboxes):
    depths = []
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        # crop = image[y0:y1, x0:x1, :]
        # plot(crop)
        crop = persons_mask[y0:y1, x0:x1]
        # plot(crop)

        depth_crop = masked_depth[y0:y1, x0:x1]
        # plot(depth_crop)
        sum = depth_crop.sum()
        n = crop.sum()
        calc_mean = sum / n

        mean_crop_ = calc_mean / 65535
        mean_crop_ = 10 * (1 - mean_crop_)
        depths.append(mean_crop_)
    return depths


def perspective_projection(points, translation, camera_center,
                           focal_length=1000, rotation=None, return_instrinsics=False):
    """
    Taken from Coherent Multiperson
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    # this is identity matrix
    rotation = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(points.device)
    # focal_length has to be fixed
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center
    # Transform points. Rotation and translation. Rotation here is identity as SMPL first rot is global
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)
    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)
    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
    if return_instrinsics:
        return projected_points[:, :, :-1], K
    else:
        return projected_points[:, :, :-1]



def weak_perspective_projection(points, translation, camera_center,
                           focal_length=1000, rotation=None):
    """
    Taken from Coherent Multiperson
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    # this is identity matrix
    rotation = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(points.device)
    # focal_length has to be fixed
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center
    # Transform points. Rotation and translation. Rotation here is identity as SMPL first rot is global
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)
    # Apply perspective distortion
    z_mean = points[:,:,-1].mean()
    z_root = points[:,14,-1]
    projected_points = points / z_root
    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
    return projected_points[:, :, :-1]


def project_joints_to_img(joints3d, img_size, translation, focal_lenght=1000,
                          return_instrinsics=False):

    projected_joints_2d = perspective_projection(joints3d,
                                                    translation,
                                                    camera_center=img_size / 2,
                                                    focal_length=focal_lenght,
                                                    return_instrinsics=return_instrinsics
                                                    )

    return projected_joints_2d

def weak_project_joints_to_img(joints3d, img_size, translation, focal_lenght=1000):

    projected_joints_2d = weak_perspective_projection(joints3d,
                                                    translation,
                                                    camera_center=img_size / 2,
                                                    focal_length=focal_lenght,
                                                    )

    return projected_joints_2d

def read_ankles(folder='./input/annots/', name='test_3djoints_0.json'):
    # these joints are in global order from CRMP so ankels=0, 5
    fpath = os.path.join(folder, name)
    keypoints = read_json(fpath)
    j3d = keypoints['joints_3d']
    trans = keypoints['translation']
    j3d = np.array(j3d)
    trans = np.array(trans)

    ankles = j3d[[0, 5]]
    ankles_translated = j3d[[0, 5]] + trans

    save_points(ankles, 'ankles_0.ply')
    return ankles, ankles_translated, trans

def read_hips(folder='./input/annots/', name='test_3djoints_0.json'):
    # these joints are in global order from CRMP so ankels=0, 5
    fpath = os.path.join(folder, name)
    keypoints = read_json(fpath)
    j3d = keypoints['joints_3d']
    trans = keypoints['translation']
    j3d = np.array(j3d)
    trans = np.array(trans)

    ankles = j3d[[2, 3]]
    ankles_translated = j3d[[2, 3]] + trans

    save_points(ankles, 'ankles_0.ply')
    return ankles, ankles_translated, trans


def read_joints(folder='./input/annots/', name='test_3djoints_0.json'):
    # these joints are in global order from CRMP so ankels=0, 5
    fpath = os.path.join(folder, name)
    keypoints = read_json(fpath)
    j3d = keypoints['joints_3d']
    trans = keypoints['translation']
    j3d = np.array(j3d)
    trans = np.array(trans)
    j3d_translated = j3d+ trans
    return j3d, j3d_translated, trans

def read_all_joints(folder='./input/annots/', names=['test_3djoints_0.json']):
    # these joints are in global order from CRMP so ankels=0, 5
    j3d_all = []
    j3d_translated_all = []
    trans_all = []
    for i, name in enumerate(names):
        fpath = os.path.join(folder, name)
        keypoints = read_json(fpath)
        j3d = keypoints['joints_3d']
        trans = keypoints['translation']
        j3d = np.array(j3d)
        trans = np.array(trans)

        j3d_translated = j3d + trans
        # save_points(j3d, 'j3d_person_%d.ply' % i)

        j3d_all.append(j3d)
        j3d_translated_all.append(j3d_translated)
        trans_all.append(trans)

    j3d_all = np.array(j3d_all)
    j3d_translated_all = np.array(j3d_translated_all)
    trans_all = np.array(trans_all)

    return j3d_all, j3d_translated_all, trans_all

# Steffen Urban, 2024

# This example demonstrates the use of pytheia to reconstruct a scene from an image sequence for gaussian splatting:
# - DISK+Lightglue for feature matching
# - Cosplace for place recognition
# - GraphMatch for two view pairing
# - todo: Alignment to gravity

import numpy as np
import cv2, os, glob, argparse, kornia
import kornia
import pytheia as pt
import torch
import natsort, time
import yaml
from yaml.loader import SafeLoader
from tqdm import tqdm

def reprojection_error(recon):

    reproj_errors = []
    for vid in recon.ViewIds():
        view = recon.View(vid)
        if view.IsEstimated():
            for tid in view.TrackIds():
                if recon.Track(tid).IsEstimated():
                    rep_pt2 = view.Camera().ProjectPoint(recon.Track(tid).Point())[1]
                    pt2 = view.GetFeature(tid).point
                    reproj_errors.append(np.linalg.norm(rep_pt2-pt2))
        
    return np.mean(np.array(reproj_errors))


def colorize_reconstruction(recon, image_path):

    color_for_track_set = set()  # Changed from list to set for faster lookup
    for idx, v_id in enumerate(recon.ViewIds()):
        view = recon.View(v_id)
        if not view.IsEstimated():
            continue

        image = cv2.imread(os.path.join(image_path, view.Name()))
        image_height, image_width, _ = image.shape

        for t_id in view.TrackIds():
            if not recon.Track(t_id).IsEstimated() or t_id in color_for_track_set:
                continue
            pt_in_img = view.GetFeature(t_id).point
            xy_int = [int(pt_in_img[0]), int(pt_in_img[1])]
            if not (0 <= xy_int[0] < image_width and 0 <= xy_int[1] < image_height):
                continue

            img_clr = image[xy_int[1], xy_int[0], :]
            recon.MutableTrack(t_id).SetColor(img_clr)
            color_for_track_set.add(t_id)  # Use add for a set

# create correspondences of keypoints locations from indexed feature matches
def correspondence_from_indexed_matches(match_indices, pts1, pts2):
    num_matches = match_indices.shape[0]
    correspondences = [
         pt.matching.FeatureCorrespondence(
            pt.sfm.Feature(pts1[match_indices[m,0],:], 0.5*np.eye(2, dtype=np.float64)), 
            pt.sfm.Feature(pts2[match_indices[m,1],:], 0.5*np.eye(2, dtype=np.float64))) for m in range(num_matches)]

    return correspondences

def draw_keypoints(img, kpts):
    img_ = np.ascontiguousarray(img.astype(np.float32))
    for id, pt in enumerate(kpts):
        p = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(img_, p, 4, (255,255,0))

    return cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)

def draw_float_matches(img1, img2, inlier_corres, si, sj):
    img1_ = np.ascontiguousarray(img1.squeeze(0).astype(np.float32))
    img2_ = np.ascontiguousarray(img2.squeeze(0).astype(np.float32))
    concat = (np.concatenate([img1_,img2_], 1) * 255).astype(np.uint8)

    for corr in inlier_corres:
        feat1 = corr.feature1.point / si
        feat2 = corr.feature2.point / sj
        p1 = (int(round(feat1[0])), int(round(feat1[1])))
        p2 = (int(round(feat2[0])), int(round(feat2[1])))
        clr = (0,255,0)
        cv2.line(concat, p1, (p2[0]+img2_.shape[1],p2[1]), clr, 1, lineType=16)
    
    return cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)

def load_img_tensor(img_path, inf_shape_max, device, dtype):
    image = kornia.io.load_image(img_path, kornia.io.ImageLoadType.RGB32)[None, ...]

    original_img_size = image.shape[3:1:-1]
    scaler = inf_shape_max / max(original_img_size)
    inf_shape_wh = (int(original_img_size[1] * scaler), 
                    int(original_img_size[0] * scaler))
    image = kornia.geometry.resize(image, inf_shape_wh, antialias=True)

    return image.to(device).to(dtype), original_img_size
import os
import sys
path = os.path.dirname(os.path.abspath(""))+"/"
sys.path.append(path)
sys.path.insert(1, path+'cellpose/')
print(path)
from skimage import io
import numpy as np
from CellAnalysis.utils import *
from CellAnalysis import evaluation
from mAP_3Dvolume import vol3d_eval_custom, vol3d_util_custom
import mAP_3Dvolume as meanap
import glob


def get_scores(pred_seg, gt_seg):
    sz_gt = np.array(gt_seg.shape)
    sz_pred = pred_seg.shape
    if np.abs((sz_gt - sz_pred)).max() > 0:
        print('Warning: size mismatch. gt: ', sz_gt, ', pred: ', sz_pred)
    sz = np.minimum(sz_gt, sz_pred)
    pred_seg = pred_seg[:sz[0], :sz[1]]
    gt_seg = gt_seg[:sz[0], :sz[1]]

    ui, uc = np.unique(pred_seg, return_counts=True)
    uc = uc[ui > 0]
    ui = ui[ui > 0]
    pred_score = np.ones([len(ui), 2], int)
    pred_score[:, 0] = ui
    pred_score[:, 1] = uc

    thres = np.fromstring('5e3, 1.5e4', sep=",")
    areaRng = np.zeros((len(thres) + 2, 2), int)
    areaRng[0, 1] = 1e10
    areaRng[-1, 1] = 1e10
    areaRng[2:, 0] = thres
    areaRng[1:-1, 1] = thres

    return pred_score, areaRng


def get_precision(pred, gt):
    pred_score, areaRng = get_scores(pred, gt)
    result_p, result_fn, pred_score_sorted = meanap.vol3d_util_custom.seg_iou2d_sorted(pred, gt, pred_score, areaRng)
    v3dEval = meanap.vol3d_eval_custom.VOL3Deval(result_p, result_fn, pred_score_sorted, output_name='map_output')
    stats = v3dEval.get_stats()
    return stats


def main():
    # file root
    file_root_0 = path + 'data/2P_functional/PC/'
    file_root_1 = path + 'data/2P_functional/CP/'
    file_root_2 = path + 'data/2P_functional/GT/'
    file_root_3 = path + 'data/2P_functional/SD/'
    file_root_4 = path + 'data/2P_functional/CP-notrain/'
    vol_1 = []
    cp = []
    vol_2 = []
    vol_3 = []
    vol_4 = []
    gt = []
    cp_notrain = []
    sd = []
    pc = []

    for name in sorted(glob.glob(file_root_0 + '*')):
        pc.append(io.imread(name))

    for name in sorted(glob.glob(file_root_1 + '*')):
        if 'mask' in name:
            cp.append(io.imread(name).astype(np.uint16))
        else:
            continue

    for name in sorted(glob.glob(file_root_2 + '*')):
        if 'mask' in name:
            gt.append(io.imread(name).astype(np.uint16))
        else:
            continue

    for name in sorted(glob.glob(file_root_3 + '*')):
        sd.append(io.imread(name))

    for name in sorted(glob.glob(file_root_4 + '*')):
        if 'mask' in name:
            cp_notrain.append(io.imread(name))
        else:
            continue

    cp = np.array(cp)
    cp_notrain = np.array(cp_notrain)
    gt = np.array(gt)
    sd = np.array(sd)
    pc = np.array(pc)

    dist_thresh = 0.5
    adc_cp = []
    adpc_cp = []
    adgc_cp = []

    adc_cp_notrain = []
    adpc_cp_notrain = []
    adgc_cp_notrain = []

    adc_sd = []
    adpc_sd = []
    adgc_sd = []

    adc_pc = []
    adpc_pc = []
    adgc_pc = []

    ap_pc = []
    ap_cp = []
    ap_sd = []
    ap_cp_notrain = []

    for i in range(np.array(gt).shape[0]):
        stats_pc = get_precision(pc[i], gt[i])
        ap_pc.append(stats_pc['Average Precision'][:, 0])
        min_adc, min_adpc, min_adgc = evaluation.average_distance_between_centroids(gt[i], pc[i], dist_thresh=dist_thresh,
                                                                                    all_stats=False, size=(0.6, 0.6))
        adc_pc.append(min_adc)
        adpc_pc.append(min_adpc)
        adgc_pc.append(min_adgc)

        stats_cp = get_precision(cp[i], gt[i])
        ap_cp.append(stats_cp['Average Precision'][:, 0])
        min_adc, min_adpc, min_adgc = evaluation.average_distance_between_centroids(gt[i], cp[i], dist_thresh=dist_thresh,
                                                                                    all_stats=False, size=(0.6, 0.6))
        adc_cp.append(min_adc)
        adpc_cp.append(min_adpc)
        adgc_cp.append(min_adgc)

        stats_sd = get_precision(sd[i], gt[i])
        ap_sd.append(stats_sd['Average Precision'][:, 0])
        min_adc, min_adpc, min_adgc = evaluation.average_distance_between_centroids(gt[i], sd[i], dist_thresh=dist_thresh,
                                                                                    all_stats=False, size=(0.6, 0.6))
        adc_sd.append(min_adc)
        adpc_sd.append(min_adpc)
        adgc_sd.append(min_adgc)

        stats_cp_notrain = get_precision(cp_notrain[i], gt[i])
        ap_cp_notrain.append(stats_cp_notrain['Average Precision'][:, 0])
        min_adc, min_adpc, min_adgc = evaluation.average_distance_between_centroids(gt[i], cp_notrain[i],
                                                                                    dist_thresh=dist_thresh,
                                                                                    all_stats=False, size=(0.6, 0.6))
        adc_cp_notrain.append(min_adc)
        adpc_cp_notrain.append(min_adpc)
        adgc_cp_notrain.append(min_adgc)

if __name__ == '__main__':
    main()
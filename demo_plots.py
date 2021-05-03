import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from mAP_3Dvolume.vol3d_eval_custom import VOL3Deval
from vol3d_util_custom import seg_iou2d_sorted, seg_iou3d_sorted, heatmap_to_score, readh5

def get_scores(pred_seg, gt_seg):
    sz_gt = np.array(gt_seg.shape)
    sz_pred = pred_seg.shape
    if np.abs((sz_gt-sz_pred)).max()>0:
        print('Warning: size mismatch. gt: ', sz_gt,', pred: ', sz_pred)
    sz = np.minimum(sz_gt, sz_pred)
    pred_seg = pred_seg[:sz[0], :sz[1]]
    gt_seg = gt_seg[:sz[0], :sz[1]]
    
    ui, uc = np.unique(pred_seg,return_counts=True)
    uc = uc[ui>0]
    ui = ui[ui>0]
    pred_score = np.ones([len(ui), 2], int)
    pred_score[:, 0] = ui
    pred_score[:, 1] = uc
    
    thres = np.fromstring('5e3, 1.5e4', sep = ",")
    areaRng = np.zeros((len(thres)+2, 2), int)
    areaRng[0, 1] = 1e10
    areaRng[-1, 1] = 1e10
    areaRng[2:, 0] = thres
    areaRng[1:-1, 1] = thres
    
    return pred_score, areaRng

def get_stats(pred, gt):
    pred_score, areaRng = get_scores(pred, gt)
    result_p, result_fn, pred_score_sorted = seg_iou3d_sorted(pred, gt, pred_score, areaRng)
    v3dEval = VOL3Deval(result_p, result_fn, pred_score_sorted, output_name='map_output')
    stats = v3dEval.get_stats()
    return stats

def main():
    file_root = 'EM/box7_testing/'

    hf_gt = h5py.File(file_root + 'gt7.h5', 'r')
    gt = hf_gt.get('main').value
    gt = gt.astype(np.uint16)
    hf_pc = h5py.File(file_root + 'pc7.h5', 'r')
    pc = hf_pc.get('main').value
    pc = pc.astype(np.uint16)
    hf_cp = h5py.File(file_root + 'cp7.h5', 'r')
    cp = hf_cp.get('main').value
    pc = pc.astype(np.uint16)
    hf_sd = h5py.File(file_root + 'sd7.h5', 'r')
    sd = hf_sd.get('main').value
    pc = pc.astype(np.uint16)
    
    stats_pc = get_stats(pc, gt)
    stats_sd = get_stats(sd, gt)
    stats_cp = get_stats(cp, gt)

    ap_pc = stats_pc['Average Precision'][:, 0]
    ap_cp = stats_cp['Average Precision'][:, 0]
    ap_sd = stats_sd['Average Precision'][:, 0]

    fig, ax = plt.subplots(figsize=(24, 7))
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    fig.suptitle('Average Precision at different IoU thresholds', fontsize=20)

    ax.plot(np.arange(0.5, 1.0, 0.05), ap_pc, linestyle='-', marker='^', color='g', label='pytoch_connectomics')
    ax.plot(np.arange(0.5, 1.0, 0.05), ap_cp, linestyle='-', marker='o', color='blue',
             label='Cellpose (pretrained only)')
    ax.plot(np.arange(0.5, 1.0, 0.05), ap_sd, linestyle='-', marker='x', color='orange', label='Stardist')
    ax.set_ylabel('Average Precision (AP)', fontsize=15)
    ax.set_xlabel('IoU Threshold', fontsize=15)
    ax.legend(fontsize=12)
    ax.set_ylim(0.0, 1.0)
    plt.show()
        
if __name__ == '__main__':
    main()
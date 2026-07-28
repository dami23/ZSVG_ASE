import torch

import ast
import pickle
import fire
import pandas as pd
from pathlib import Path
from anchors import IoU_values

def evaluate(pred_file, gt_file, **kwargs):
    acc_iou_thresh = kwargs.get('acc_iou_thresh', 0.5)
    pred_file = Path(pred_file)
    if not pred_file.exists():
        assert 'num_gpus' in kwargs
        num_gpus = kwargs['num_gpus']
        pred_files_to_use = [pred_file.parent /
                             f'{r}_{pred_file.name}' for r in range(num_gpus)]
        assert all([p.exists() for p in pred_files_to_use])
        out_preds = []
        for pf in pred_files_to_use:
            tmp = pickle.load(open(pf, 'rb'))
            assert isinstance(tmp, list)
            out_preds += tmp
        pickle.dump(out_preds, pred_file.open('wb'))

    predictions = pickle.load(open(pred_file, 'rb'))
    gt_annot = pd.read_csv(gt_file)
    # gt_annot = gt_annot.iloc[:len(predictions)]
    gt_annot['bbox'] = gt_annot.bbox.apply(lambda x: ast.literal_eval(x))

    # assert len(predictions) == len(gt_annot)
    corr = 0
    tot = 0
    inds_used = set()
    for p in predictions:
        ind = int(p['id'])
        if ind not in inds_used:
            inds_used.add(ind)
            annot = gt_annot.iloc[ind]
            gt_box = torch.tensor(annot.bbox)
            pred_box = torch.tensor(p['pred_boxes'])

            iou = IoU_values(pred_box[None, :], gt_box[None, :])
            if iou > acc_iou_thresh:
                corr += 1
            tot += 1
    return corr/tot, corr, tot


if __name__ == '__main__':
    fire.Fire(evaluate)

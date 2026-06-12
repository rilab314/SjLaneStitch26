import os
import sys
import json
import glob
import tempfile

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from evaluator import load_json, ann_to_mask, to_label_index_image, evaluate_segm_pred_metrics
from util import find_best_pred_json_path, find_model_path


class Table2Builder:
    def __init__(self, table1_csv_path, save_path):
        self.table1_csv_path = table1_csv_path
        self.save_path = save_path
        self.gt_json = cfg.COCO_MERGED_ANNO_PATH
        self.label_dir = os.path.join(cfg.DATASET_PATH, 'annotations', 'validation')

    def build(self):
        model_name, merge_count, pred_json_path = find_best_pred_json_path(self.table1_csv_path)
        print('pred_json_path: ', pred_json_path)
        gt_counts, pred_counts = self._count_objects(pred_json_path)
        ap20_dict = self._evaluate_ap20_per_class(pred_json_path)
        seg_iou_dict = self._evaluate_segm_iou_per_class(model_name)
        miou_dict = self._evaluate_miou_per_class(pred_json_path)
        result = self._build_result_df(ap20_dict, seg_iou_dict, miou_dict, gt_counts, pred_counts)
        self._save(result)
        self._verify(result, model_name, merge_count)


    def _count_objects(self, pred_json_path):
        print("\n개체 수 집계 중...")
        gt_data = self._load_filtered_gt()
        pred_data = load_json(pred_json_path)
        preds = pred_data['annotations'] if isinstance(pred_data, dict) and 'annotations' in pred_data else pred_data

        gt_counts = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}
        pred_counts = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}

        for ann in gt_data.get('annotations', []):
            cid = ann.get('category_id')
            if cid in gt_counts:
                gt_counts[cid] += 1

        for ann in preds:
            cid = ann.get('category_id')
            if cid in pred_counts:
                pred_counts[cid] += 1

        return gt_counts, pred_counts

    def _evaluate_ap20_per_class(self, pred_json_path):
        print("\nPer-class AP20 평가 중...")
        gt_data = self._load_filtered_gt()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(gt_data, tmp)
            tmp_path = tmp.name

        coco_gt = COCO(tmp_path)
        pred_data = [d for d in load_json(pred_json_path) if d.get('category_id') not in cfg.EXCLUDE_IDS]
        coco_pred = coco_gt.loadRes(pred_data)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.catIds = cfg.EVAL_CLASS_IDS
        coco_eval.params.iouThrs = np.array([0.20], dtype=np.float32)
        coco_eval.evaluate()
        coco_eval.accumulate()
        os.remove(tmp_path)

        ap20_dict = {
            int(cid): self._extract_ap(coco_eval, idx)
            for idx, cid in enumerate(coco_eval.params.catIds)
        }
        return ap20_dict

    def _load_filtered_gt(self):
        gt_data = load_json(self.gt_json)
        gt_data['annotations'] = [
            a for a in gt_data.get('annotations', [])
            if a.get('category_id') not in cfg.EXCLUDE_IDS
        ]
        for i, ann in enumerate(gt_data['annotations']):
            ann.setdefault('id', i + 1)
            ann.setdefault('iscrowd', 0)
            if 'area' not in ann and 'segmentation' in ann:
                ann['area'] = float(
                    maskUtils.area(ann['segmentation'])
                    if isinstance(ann['segmentation'], dict) else 0.0
                )
        return gt_data

    def _extract_ap(self, coco_eval, class_idx):
        p = coco_eval.eval['precision'][0, :, class_idx, 0, -1]
        p = p[p > -1]
        ap = float(np.mean(p)) if p.size > 0 else 0.0
        return ap

    def _evaluate_segm_iou_per_class(self, model_name):
        print("\nPer-class segmentation IoU 평가 중 (알고리즘 처리 전 순수 예측)...")
        model_path = find_model_path(model_name if model_name else 'internimage_large')
        metrics = evaluate_segm_pred_metrics(model_path, self.label_dir)
        return {int(cid): iou for cid, iou in metrics['per_class_iou'].items()}

    def _evaluate_miou_per_class(self, pred_json_path):
        print("\nPer-class mIoU 평가 중...")
        data = load_json(pred_json_path)
        anns = data['annotations'] if isinstance(data, dict) else data
        ann_idx = self._build_ann_index(anns)

        intersections = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}
        unions = {cid: 0 for cid in cfg.EVAL_CLASS_IDS}

        for file in tqdm(glob.glob(os.path.join(self.label_dir, '*.png')), desc="Per-class mIoU"):
            grtr_label = to_label_index_image(cv2.imread(file, cv2.IMREAD_UNCHANGED), True)
            if grtr_label is None:
                continue
            h, w = grtr_label.shape
            pred_label = self._build_pred_label(ann_idx, h, w, file)
            for cid in cfg.EVAL_CLASS_IDS:
                intersections[cid] += int(np.sum((grtr_label == cid) & (pred_label == cid)))
                unions[cid] += int(np.sum((grtr_label == cid) | (pred_label == cid)))

        miou_dict = {
            cid: (intersections[cid] / unions[cid] if unions[cid] > 0 else 0.0)
            for cid in cfg.EVAL_CLASS_IDS
        }
        return miou_dict

    def _build_ann_index(self, anns):
        ann_idx = {}
        for a in anns:
            if int(a.get('category_id', 0)) in cfg.EXCLUDE_IDS:
                continue
            ann_idx.setdefault(str(a.get('image_id')), []).append(a)
        return ann_idx

    def _build_pred_label(self, ann_idx, h, w, file):
        pred_label = np.zeros((h, w), dtype=np.int32)
        for ann in ann_idx.get(os.path.basename(file).replace('.png', ''), []):
            pred_label[ann_to_mask(ann, h, w) > 0] = int(ann.get('category_id', 0))
        return pred_label

    def _build_result_df(self, ap20_dict, seg_iou_dict, miou_dict, gt_counts, pred_counts):
        rows = [
            {'class_name': cfg.ID2NAME.get(cid, str(cid)),
             'gt_count': gt_counts.get(cid, 0),
             'pred_count': pred_counts.get(cid, 0),
             'seg_IoU': seg_iou_dict.get(cid, 0.0),
             'AP20': ap20_dict.get(cid, 0.0),
             'mIoU': miou_dict.get(cid, 0.0)}
            for cid in cfg.EVAL_CLASS_IDS
        ]
        df = pd.DataFrame(rows, columns=['class_name', 'gt_count', 'pred_count', 'seg_IoU', 'AP20', 'mIoU'])
        return df

    def _save(self, result):
        result[['seg_IoU', 'AP20', 'mIoU']] = result[['seg_IoU', 'AP20', 'mIoU']].round(4)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        result.to_csv(self.save_path, index=False, encoding='utf-8')
        print(f"\nTable 2 saved to: {self.save_path}")
        print(result.to_string(index=False))

    def _verify(self, result, model_name, merge_count):
        df1 = pd.read_csv(self.table1_csv_path)
        t1_row = df1[(df1['model_name'] == model_name) & (df1['merge_count'] == merge_count)]
        if t1_row.empty:
            print(f"\n검증 실패: table_1.csv에서 ({model_name}, merge_count={merge_count}) 행을 찾을 수 없음")
            return

        avg_ap20 = result['AP20'].mean()
        avg_miou = result['mIoU'].mean()
        t1_ap20 = float(t1_row['AP20'].values[0])
        t1_miou = float(t1_row['mIoU'].values[0])

        print(f"\n{'검증: Table 2 (클래스 평균) vs Table 1':^80}")
        print(f"{'Metric':<10} | {'Table 2 (Mean)':<20} | {'Table 1':^20} | {'Diff':<15}")
        print("-" * 75)
        print(f"{'AP20':<10} | {avg_ap20:<20.6f} | {t1_ap20:<20.6f} | {abs(avg_ap20 - t1_ap20):<15.6e}")
        print(f"{'mIoU':<10} | {avg_miou:<20.6f} | {t1_miou:<20.6f} | {abs(avg_miou - t1_miou):<15.6e}")

        if abs(avg_ap20 - t1_ap20) < 1e-4 and abs(avg_miou - t1_miou) < 1e-4:
            print("\n검증 성공: Table 2 클래스 평균이 Table 1 결과와 일치합니다.")
        else:
            print("\n검증 경고: 유의미한 차이가 감지되었습니다.")


def main():
    table1_csv_path = os.path.join(cfg.RESULT_PATH, 'Tables', 'table_1.csv')
    save_path = os.path.join(cfg.RESULT_PATH, 'Tables', 'table_2.csv')
    Table2Builder(table1_csv_path, save_path).build()


if __name__ == '__main__':
    main()

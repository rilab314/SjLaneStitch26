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
from evaluator import load_json, ann_to_mask, to_label_index_image
from util import find_best_pred_json_path


class Table2Builder:
    def __init__(self, table1_csv_path, save_path, supp_save_path):
        self.table1_csv_path = table1_csv_path
        self.save_path = save_path
        self.supp_save_path = supp_save_path
        self.gt_json = cfg.COCO_MERGED_ANNO_PATH
        self.label_dir = os.path.join(cfg.DATASET_PATH, 'annotations', 'validation')

    def build(self):
        model_name, merge_count, pred_json_path = find_best_pred_json_path(self.table1_csv_path)
        print('pred_json_path: ', pred_json_path)
        gt_counts, pred_counts = self._count_objects(pred_json_path)
        ap20_dict = self._evaluate_ap20_per_class(pred_json_path)
        miou_dict = self._evaluate_miou_per_class(pred_json_path)
        result = self._build_result_df(ap20_dict, miou_dict, gt_counts, pred_counts)
        self._save(result)
        self._verify(result, model_name, merge_count)
        supp = self._compute_supp_metrics(pred_json_path)
        self._save_supp(supp)


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

    def _build_result_df(self, ap20_dict, miou_dict, gt_counts, pred_counts):
        rows = [
            {'class_name': cfg.ID2NAME.get(cid, str(cid)),
             'gt_count': gt_counts.get(cid, 0),
             'pred_count': pred_counts.get(cid, 0),
             'mIoU': miou_dict.get(cid, 0.0),
             'AP20': ap20_dict.get(cid, 0.0)}
            for cid in cfg.EVAL_CLASS_IDS
        ]
        df = pd.DataFrame(rows, columns=['class_name', 'gt_count', 'pred_count', 'mIoU', 'AP20'])
        return df

    def _save(self, result):
        metric_cols = ['mIoU', 'AP20']
        result[metric_cols] = (result[metric_cols] * 100).round(2)  # % 단위로 변환
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        result.to_csv(self.save_path, index=False, encoding='utf-8')
        print(f"\nTable 2 saved to: {self.save_path}")
        print(result.to_string(index=False))

    # ------------------------------------------------------------------
    # table_3: 클래스별 심화 지표 (mIoU vs AP20 괴리 분석)
    # ------------------------------------------------------------------
    def _compute_supp_metrics(self, pred_json_path):
        print("\ntable_3 심화 지표 계산 중 (클래스별 인스턴스 매칭 분석)...")
        gt_idx = self._group_by_image_class(self._load_filtered_gt()['annotations'])
        pred_data = load_json(pred_json_path)
        pred_anns = pred_data['annotations'] if isinstance(pred_data, dict) else pred_data
        pred_idx = self._group_by_image_class(pred_anns)

        acc = {cid: self._new_acc() for cid in cfg.EVAL_CLASS_IDS}
        image_ids = set(gt_idx) | set(pred_idx)
        for img_id in tqdm(image_ids, desc="Supp metrics"):
            for cid in cfg.EVAL_CLASS_IDS:
                gts = gt_idx.get(img_id, {}).get(cid, [])
                prs = pred_idx.get(img_id, {}).get(cid, [])
                self._accumulate_class(acc[cid], gts, prs)
        return {cid: self._finalize_acc(acc[cid]) for cid in cfg.EVAL_CLASS_IDS}

    def _group_by_image_class(self, anns):
        idx = {}
        for a in anns:
            cid = int(a.get('category_id', 0))
            if cid in cfg.EXCLUDE_IDS:
                continue
            idx.setdefault(str(a.get('image_id')), {}).setdefault(cid, []).append(a)
        return idx

    def _new_acc(self):
        return {'n_gt': 0, 'n_pred': 0, 'M': 0, 'matched_iou_sum': 0.0,
                'gt_area_total': 0.0, 'pred_area_total': 0.0,
                'near_miss_gt': 0, 'near_miss_pix': 0.0,
                'merge_sum': 0, 'frag_sum': 0, 'fp_inst': 0, 'fp_pix': 0.0}

    def _accumulate_class(self, acc, gts, prs):
        nG, nP = len(gts), len(prs)
        acc['n_gt'] += nG
        acc['n_pred'] += nP
        gt_segs = [g['segmentation'] for g in gts]
        pr_segs = [p['segmentation'] for p in prs]
        gt_areas = [float(maskUtils.area(s)) for s in gt_segs]
        pr_areas = [float(maskUtils.area(s)) for s in pr_segs]
        acc['gt_area_total'] += sum(gt_areas)
        acc['pred_area_total'] += sum(pr_areas)

        # X[i, j] = IoU(예측 i, GT j) — 같은 이미지·클래스 내에서만 계산
        if nP and nG:
            X = np.asarray(maskUtils.iou(pr_segs, gt_segs, [0] * nG),
                           dtype=np.float64).reshape(nP, nG)
        else:
            X = np.zeros((nP, nG), dtype=np.float64)

        # GT 측 지표: near-miss(임계 미달 겹침), fragmentation(분할)
        for j in range(nG):
            col = X[:, j]
            bj = float(col.max()) if nP else 0.0
            if 0.0 < bj < 0.2:
                acc['near_miss_gt'] += 1
                acc['near_miss_pix'] += gt_areas[j]
            acc['frag_sum'] += int((col > 0).sum())

        # 예측 측 지표: merge(병합), pure-FP(GT와 전혀 안 겹침)
        for i in range(nP):
            row = X[i, :]
            acc['merge_sum'] += int((row > 0).sum())
            if (float(row.max()) if nG else 0.0) == 0.0:
                acc['fp_inst'] += 1
                acc['fp_pix'] += pr_areas[i]

        # IoU>=0.2 에서 1:1 greedy 매칭 (높은 IoU 우선)
        for v in self._greedy_match(X):
            acc['M'] += 1
            acc['matched_iou_sum'] += v

    def _greedy_match(self, X, thr=0.2):
        nP, nG = X.shape
        pairs = [(X[i, j], i, j) for i in range(nP) for j in range(nG) if X[i, j] >= thr]
        pairs.sort(reverse=True)
        used_p, used_g, ious = set(), set(), []
        for v, i, j in pairs:
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            ious.append(v)
        return ious

    def _finalize_acc(self, a):
        nP, nG, M = a['n_pred'], a['n_gt'], a['M']
        precision = M / nP if nP else 0.0
        recall = M / nG if nG else 0.0
        return {
            'precision': precision,                                          # 1. M / pred_count
            'recall': recall,                                                # 2. M / gt_count
            'ap20_check': (M * M) / (nP * nG) if nP and nG else 0.0,         # 3. P x R 검증값
            'count_ratio': nP / nG if nG else 0.0,                           # 4. pred_count / gt_count
            'near_miss_gt': a['near_miss_gt'] / nG if nG else 0.0,           # 5. 0<best IoU<0.2 GT 비율
            'near_miss_pix': a['near_miss_pix'] / a['gt_area_total']         # 6. near-miss GT 픽셀 질량비
                             if a['gt_area_total'] else 0.0,
            'merge_ratio': a['merge_sum'] / nP if nP else 0.0,               # 7. 예측당 겹친 GT 평균 수
            'frag_ratio': a['frag_sum'] / nG if nG else 0.0,                 # 8. GT당 겹친 예측 평균 수
            'miou_match': a['matched_iou_sum'] / M if M else 0.0,            # 9. 매칭쌍 평균 IoU
            'fp_inst': a['fp_inst'] / nP if nP else 0.0,                     # 10a. pure-FP 인스턴스 비율
            'fp_pix': a['fp_pix'] / a['pred_area_total']                     # 10b. pure-FP 픽셀 비율
                      if a['pred_area_total'] else 0.0,
        }

    def _save_supp(self, supp):
        pct_cols = ['precision', 'recall', 'ap20_check', 'near_miss_gt',
                    'near_miss_pix', 'miou_match', 'fp_inst', 'fp_pix']
        ratio_cols = ['count_ratio', 'merge_ratio', 'frag_ratio']
        cols = ['class_name', 'precision', 'recall', 'ap20_check', 'count_ratio',
                'near_miss_gt', 'near_miss_pix', 'merge_ratio', 'frag_ratio',
                'miou_match', 'fp_inst', 'fp_pix']
        rows = []
        for cid in cfg.EVAL_CLASS_IDS:
            m = supp[cid]
            row = {'class_name': cfg.ID2NAME.get(cid, str(cid))}
            row.update({k: round(m[k] * 100, 2) for k in pct_cols})       # 비율형 -> % 단위
            row.update({k: round(m[k], 3) for k in ratio_cols})           # 배율형 -> 그대로
            rows.append(row)
        df = pd.DataFrame(rows, columns=cols)
        os.makedirs(os.path.dirname(self.supp_save_path), exist_ok=True)
        df.to_csv(self.supp_save_path, index=False, encoding='utf-8')
        print(f"\nTable 3 saved to: {self.supp_save_path}")
        print(df.to_string(index=False))

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

        # % 단위 기준 허용 오차 0.01%p (반올림 오차 수준)
        if abs(avg_ap20 - t1_ap20) < 1e-2 and abs(avg_miou - t1_miou) < 1e-2:
            print("\n검증 성공: Table 2 클래스 평균이 Table 1 결과와 일치합니다.")
        else:
            print("\n검증 경고: 유의미한 차이가 감지되었습니다.")


def main():
    table1_csv_path = os.path.join(cfg.RESULT_PATH, 'Tables', 'table_1.csv')
    save_path = os.path.join(cfg.RESULT_PATH, 'Tables', 'table_2.csv')
    supp_save_path = os.path.join(cfg.RESULT_PATH, 'Tables', 'table_3.csv')
    Table2Builder(table1_csv_path, save_path, supp_save_path).build()


if __name__ == '__main__':
    main()

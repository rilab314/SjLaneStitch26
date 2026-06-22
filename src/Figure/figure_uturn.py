import os
import sys

import cv2
import numpy as np
from pycocotools import mask as maskUtils

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import src.config as cfg
from src.util import find_best_pred_json_path, load_json, group_annotations_by_image

UTURN_ID = 2          # u_turn_zone_line
IOU_THR = 0.2         # AP20 / 매칭 임계값 (table_2 와 동일)
ALPHA = 0.55          # 마스크 오버레이 투명도

# GT 인스턴스 테두리 색 (BGR)
COL_MATCHED = (0, 220, 0)        # 초록: 매칭 성공
COL_MISS_MERGE = (0, 165, 255)   # 주황: IoU>=0.2 인 예측이 있으나 1:1 매칭에서 형제 GT에 뺏김 (병합 손실)
COL_MISS_NEAR = (0, 255, 255)    # 노랑: 0<IoU<0.2 (임계 미달)
COL_MISS_PURE = (0, 0, 255)      # 빨강: 겹치는 예측이 전혀 없음

WIN = "u_turn miss viewer  (any key=next, s=auto toggle, q/esc=quit)"


def decode_mask(ann):
    m = maskUtils.decode(ann['segmentation'])
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def distinct_colors(n):
    """HSV 색상환을 균등 분할해 인스턴스별로 구분되는 BGR 색을 만든다."""
    colors = []
    for i in range(max(n, 1)):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def iou_matrix(pred_segs, gt_segs):
    if not pred_segs or not gt_segs:
        return np.zeros((len(pred_segs), len(gt_segs)), dtype=np.float64)
    X = maskUtils.iou(pred_segs, gt_segs, [0] * len(gt_segs))
    return np.asarray(X, dtype=np.float64).reshape(len(pred_segs), len(gt_segs))


def greedy_match(X, thr=IOU_THR):
    """IoU 높은 쌍부터 1:1 greedy 매칭. 매칭된 GT 인덱스 집합과 pred->gt 매핑 반환."""
    nP, nG = X.shape
    pairs = sorted(
        ((X[i, j], i, j) for i in range(nP) for j in range(nG) if X[i, j] >= thr),
        reverse=True,
    )
    used_p, used_g, gt2pred = set(), set(), {}
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        gt2pred[j] = i
    return used_g, gt2pred


def classify_gt(X, matched_gt):
    """매칭 안 된 각 GT 를 miss 유형으로 분류. 0=matched,1=merge,2=near,3=pure."""
    nP, nG = X.shape
    cats = []
    for j in range(nG):
        if j in matched_gt:
            cats.append(0)
            continue
        best = float(X[:, j].max()) if nP else 0.0
        if best >= IOU_THR:
            cats.append(1)   # 매칭 가능한 예측이 있었으나 형제 GT 에 뺏김 (병합 손실)
        elif best > 0.0:
            cats.append(2)   # near-miss
        else:
            cats.append(3)   # pure miss
    return cats


def overlay(base, mask, color, alpha=ALPHA):
    sel = mask > 0
    base[sel] = (base[sel] * (1 - alpha) + np.array(color, dtype=np.float64) * alpha).astype(np.uint8)


def draw_border(img, mask, color, thick=2):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thick)


def banner(img, lines):
    """이미지 상단에 반투명 검은 띠 + 텍스트."""
    h = 22 * len(lines) + 10
    img[0:h] = (img[0:h] * 0.25).astype(np.uint8)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (8, 20 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def render(base, gt_anns, pred_anns, X, matched_gt, cats):
    """좌: GT(인스턴스별 색 + miss 유형 테두리), 우: 예측(인스턴스별 색)."""
    gt_cols = distinct_colors(len(gt_anns))
    pred_cols = distinct_colors(len(pred_anns))
    border_cols = [COL_MATCHED, COL_MISS_MERGE, COL_MISS_NEAR, COL_MISS_PURE]

    panel_gt = base.copy()
    for j, ann in enumerate(gt_anns):
        m = decode_mask(ann)
        overlay(panel_gt, m, gt_cols[j])
        draw_border(panel_gt, m, border_cols[cats[j]], thick=2)

    panel_pred = base.copy()
    for i, ann in enumerate(pred_anns):
        m = decode_mask(ann)
        overlay(panel_pred, m, pred_cols[i])
        draw_border(panel_pred, m, (255, 255, 255), thick=1)

    nG, nP = len(gt_anns), len(pred_anns)
    n_match = len(matched_gt)
    n_merge = cats.count(1)
    n_near = cats.count(2)
    n_pure = cats.count(3)
    cover = int(np.sum(X.max(axis=0) >= IOU_THR)) if nP and nG else 0

    banner(panel_gt, [
        f"GT u_turn: {nG}  matched(green): {n_match}  recall: {100*n_match/nG:.1f}%",
        f"miss-merge(orange): {n_merge}  near(yellow): {n_near}  pure(red): {n_pure}",
        f"IoU>=0.2 로 덮인 GT(=many-to-one recall): {cover} ({100*cover/nG:.1f}%)",
    ])
    banner(panel_pred, [
        f"Pred u_turn: {nP}  (한 예측당 겹친 GT 평균: {X.sum() and (X>0).sum(0).mean():.2f})",
        f"pred/gt count ratio: {nP/nG:.2f}",
    ])

    gap = np.full((base.shape[0], 16, 3), 255, dtype=np.uint8)
    return np.hstack([panel_gt, gap, panel_pred])


def wait_key(continuous):
    return cv2.waitKey(200 if continuous else 0) & 0xFF


def main():
    csv_path = os.path.join(cfg.RESULT_PATH, 'Tables', 'table_1.csv')
    _, _, pred_json_path = find_best_pred_json_path(csv_path)
    print("pred_json_path:", pred_json_path)

    gt_data = load_json(cfg.COCO_ANNO_PATH)
    pred_anns_all = load_json(pred_json_path)
    pred_anns_all = pred_anns_all['annotations'] if isinstance(pred_anns_all, dict) else pred_anns_all

    gt_map = group_annotations_by_image(
        [a for a in gt_data['annotations'] if a.get('category_id') == UTURN_ID])
    pred_map = group_annotations_by_image(
        [a for a in pred_anns_all if a.get('category_id') == UTURN_ID])

    img_dir = os.path.join(cfg.DATASET_PATH, 'images', 'validation')

    # 전체 통계 누적기
    agg = dict(nGT=0, nPred=0, M=0, cover=0, miss_merge=0, miss_near=0, miss_pure=0,
               imgs_with_gt=0, imgs_with_miss=0, multi_gt_preds=0, pred_overlap_sum=0,
               area_match=[], area_merge=[], area_near=[], area_pure=[])

    missed_cases = []  # (file_name, img_id) 시각화 대상

    for img_info in gt_data['images']:
        img_id = img_info['id']
        gt_anns = gt_map.get(img_id, [])
        if not gt_anns:
            continue
        pred_anns = pred_map.get(img_id, [])
        gt_segs = [a['segmentation'] for a in gt_anns]
        pred_segs = [a['segmentation'] for a in pred_anns]
        X = iou_matrix(pred_segs, gt_segs)
        matched_gt, _ = greedy_match(X)
        cats = classify_gt(X, matched_gt)

        agg['imgs_with_gt'] += 1
        agg['nGT'] += len(gt_anns)
        agg['nPred'] += len(pred_anns)
        agg['M'] += len(matched_gt)
        if len(pred_anns) and len(gt_anns):
            agg['cover'] += int(np.sum(X.max(axis=0) >= IOU_THR))
            overlaps = (X > 0).sum(axis=1)            # 각 예측이 겹친 GT 수
            agg['multi_gt_preds'] += int(np.sum(overlaps >= 2))
            agg['pred_overlap_sum'] += int(overlaps.sum())
        agg['miss_merge'] += cats.count(1)
        agg['miss_near'] += cats.count(2)
        agg['miss_pure'] += cats.count(3)
        for j, a in enumerate(gt_anns):
            ar = float(maskUtils.area(a['segmentation']))
            [agg['area_match'], agg['area_merge'], agg['area_near'], agg['area_pure']][cats[j]].append(ar)

        if any(c != 0 for c in cats):
            agg['imgs_with_miss'] += 1
            missed_cases.append((img_info['file_name'], img_id))

    print_stats(agg)

    # ---- 시각화: miss 가 있는 이미지만 표시 ----
    print(f"\nmiss 케이스 {len(missed_cases)}장 표시. (아무 키=다음, s=자동재생 토글, q/esc=종료)")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    continuous = False
    for file_name, img_id in missed_cases:
        base = cv2.imread(os.path.join(img_dir, file_name))
        if base is None:
            continue
        gt_anns = gt_map.get(img_id, [])
        pred_anns = pred_map.get(img_id, [])
        X = iou_matrix([a['segmentation'] for a in pred_anns], [a['segmentation'] for a in gt_anns])
        matched_gt, _ = greedy_match(X)
        cats = classify_gt(X, matched_gt)
        canvas = render(base, gt_anns, pred_anns, X, matched_gt, cats)

        advance = False
        while not advance:
            cv2.imshow(WIN, canvas)
            key = wait_key(continuous)
            if key in (ord('q'), 27):
                cv2.destroyAllWindows()
                return
            if key == ord('s'):
                continuous = not continuous
                advance = continuous       # 자동재생 켜질 땐 바로 다음, 꺼질 땐 현재 화면 유지
            elif continuous:
                advance = True             # 타임아웃(키 없음) -> 자동 진행
            else:
                advance = True             # 단계 모드에서 아무 키 -> 다음
    cv2.destroyAllWindows()


def print_stats(a):
    nGT, nPred, M = a['nGT'], a['nPred'], a['M']
    cover = a['cover']
    sep = "=" * 64

    def mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    print("\n" + sep)
    print("u_turn_zone_line 분석 통계 (best prediction, IoU>=0.2)")
    print(sep)
    print(f"u_turn GT 가 있는 이미지: {a['imgs_with_gt']}장  /  miss 발생: {a['imgs_with_miss']}장")
    print(f"GT 총 개수      : {nGT}")
    print(f"예측 총 개수    : {nPred}")
    print(f"pred/gt count ratio : {nPred/nGT:.3f}   (1 보다 작음 = 예측이 GT 보다 적음)")
    print(f"한 예측당 겹친 GT 평균(merge_ratio): {a['pred_overlap_sum']/nPred:.3f}")
    print(f"2개 이상 GT 를 덮은 예측 수: {a['multi_gt_preds']} "
          f"({100*a['multi_gt_preds']/nPred:.1f}% of preds)")
    print("-" * 64)
    print(f"1:1 매칭 성공 M : {M}   -> recall = {100*M/nGT:.2f}%  precision = {100*M/nPred:.2f}%")
    print(f"IoU>=0.2 로 덮인 GT(many-to-one 가정 recall): {cover} = {100*cover/nGT:.2f}%")
    print(f"  └ 병합 때문에 잃은 recall (cover - M): {cover-M} = {100*(cover-M)/nGT:.2f}%p")
    print("-" * 64)
    print("매칭 실패 GT 분류:")
    print(f"  병합 손실(IoU>=0.2 인데 형제 GT 에 뺏김): {a['miss_merge']} ({100*a['miss_merge']/nGT:.1f}%)")
    print(f"  near-miss (0<IoU<0.2)                  : {a['miss_near']} ({100*a['miss_near']/nGT:.1f}%)")
    print(f"  pure-miss (겹치는 예측 없음)            : {a['miss_pure']} ({100*a['miss_pure']/nGT:.1f}%)")
    print("-" * 64)
    print("GT 인스턴스 평균 면적(px):")
    print(f"  matched   : {mean(a['area_match']):8.1f}")
    print(f"  miss-merge: {mean(a['area_merge']):8.1f}")
    print(f"  miss-near : {mean(a['area_near']):8.1f}")
    print(f"  miss-pure : {mean(a['area_pure']):8.1f}")
    print(sep)
    print("해석: count_ratio<1 + merge_ratio>>1 + (병합 손실 비중 큼) 이면,")
    print("      GT 가 잘게 쪼개져 있고 모델은 이를 하나의 blob 으로 합쳐 예측 ->")
    print("      픽셀 단위로는 잘 덮어(mIoU 높음) 보이지만 1:1 매칭에서 GT 다수가")
    print("      누락되어 recall/AP20 이 낮게 나오는 것이 원인.")
    print(sep)


if __name__ == "__main__":
    main()

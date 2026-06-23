import os
import glob

import cv2
import numpy as np
import json
import copy
from pycocotools import mask as maskUtils
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass
from tqdm import tqdm


from show_imgs import ImageShow
import config as cfg
import polyline_merge as pm


@dataclass
class LineString:
    id: int
    peak: Tuple[int, int]
    class_id: int
    points: np.ndarray = None  # 원본 선상의 샘플링된 점들 (N,2)
    ext_points: np.ndarray = None  # 양쪽으로 확장된 선상의 점들 ((N+M),2)
    src_range: Tuple[int, int] = None  # ext_points 내에서 원래 points가 차지하는 인덱스 범위
    length: float = 0  # 선의 길이 (유클리드 누적거리)


class LineStringDetector:
    id_offset = 10  # peak ID의 최소 오프셋
    overlap_thresh = 2  # 겹치는 픽셀 수
    short_length = 30
    num_merges = 3

    # --- 공유 모듈(polyline_merge) 기반 정리/병합 파라미터 (merge_annotation과 동일 기조) ---
    trim_class_id = 1        # 겹침 trim 적용 클래스 (center_line)
    dup_dist = 3.0           # 복제선 판정 점-본체 거리
    dup_ratio = 0.8          # 복제선 판정 비율
    trim_step = 3.0          # trim 대상 재샘플 간격
    overlap_dist = 6.0       # 갈라짐 강임계(px)
    overlap_low = 3.0        # 갈라짐 약임계(px, hysteresis)
    min_free_len = 20.0      # trim 후 남길 최소 조각 길이(px)
    bridge_gap = 10.0        # trim 짧은 겹침 단절 브리징(px)
    parallel_overlap = 0.5   # 평행 본체 종축 겹침 임계
    parallel_lateral = 30.0  # 평행 본체 최대 측면 간격(px)
    turn_penalty = 3.0       # 샘플링 시 다음점 선택의 곡률 패널티. 분기에서 곧게 잇는 쪽을 선호(0=거리만)
    smooth_window = 5        # 병합 후 점 스무딩 이동평균 창 크기(점)
    smooth_iters = 1         # 스무딩 반복 횟수
    residual_pass = True     # 1차 추출 후 남은 seg 영역에서 한 번 더 추출 (이중선 반대쪽 복원)
    residual_remove_width = 7 # 잔여 추출 시 1차 선 자취를 지우는 두께(px)

    def __init__(self, data_path: str, pred_path: str, result_path: str, thickness: int = 3, sample_stride: int = 10, extend_len: int = 20, visualize: bool = True):
        self.thickness = thickness
        self.sample_stride = sample_stride
        self.extend_len = extend_len
        self._data_path = data_path
        self._pred_path = pred_path
        self._result_path = result_path
        self._visualize = visualize  # False면 창 표시·시각화 콜라주 생략 (성능평가용 고속 모드)
        self._img_shape = (100, 100)
        self._palette = [info['color'][::-1] for info in cfg.METAINFO]
        # print('palette:', self._palette)
        self._id_count = 0
        self._imshow_base = ImageShow('base images', columns=3, scale=0.8, enabled=visualize)
        self._imshow_proc = ImageShow('processing images', columns=3, scale=0.8, enabled=visualize)
        self._imshow_save = ImageShow('save images', columns=3, scale=0.5, enabled=visualize)
        self._exclude_classes = [0]
        # self.figure_path = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector(copy)/ade20k/result/Figure'
        assert os.path.exists(self._data_path), f"data_path: {self._data_path} is not exists"
        print("make result path: ", self._result_path)
        os.makedirs(self._result_path, exist_ok=True)

    def detect_lines(self, image_ids=None, desc=None):
        file_list = sorted(glob.glob(os.path.join(self._data_path, 'images', 'validation', '*.png')))
        if image_ids is not None:  # 특정 이미지 부분집합만 처리 (실험/비교용)
            keep = set(image_ids)
            file_list = [f for f in file_list if os.path.basename(f)[:-4] in keep]
        result_jsons = [[] for _ in range(self.num_merges + 1)]  # index 0=origin, 1~3=merge1~3

        pbar = tqdm(enumerate(file_list), total=len(file_list), desc=desc or 'frames')
        for i, file_name in pbar:
            # if i > 10:
            #     break
            pbar.set_postfix_str(os.path.basename(file_name))
            image, pred_img, anno_img = self._read_image(file_name)
            self._img_shape = image.shape[:2]
            self._id_count = self.id_offset
            image_id = os.path.basename(file_name).replace('.png', '')

            lines, line_img = self.extract_lines(pred_img, file_name)
            lines = self._clean_lines(lines)  # 복제 제거 + 평행 겹침 trim (지그재그 방지)
            if self.residual_pass:
                # 1차 선들이 덮지 못한 seg 영역에서 한 번 더 추출 (안전지대를 두고
                # 갈라졌다 만나는 이중 중앙선의 반대쪽 등 누락 선 복원)
                res, _ = self.extract_lines(self._residual_pred(pred_img, lines), file_name)
                lines = self._combine_dedup(lines + self._clean_lines(res))
            result_jsons[0] += self.convert_to_json(self._smoothed_copies(lines), image_id)
            images_to_save = {'src_img': image, 'anno_img': anno_img, 'pred_img': pred_img, 'origin': line_img}

            for n in range(1, self.num_merges + 1):
                lines, line_img = self.merge_lines(lines, n - 1)
                # 병합 완료 후 출력 단계에서 점들을 스무딩 (병합 자체는 원본 점으로 진행)
                result_jsons[n] += self.convert_to_json(self._smoothed_copies(lines), image_id)
                images_to_save[f'merge{n}'] = line_img

            if self._visualize:
                self._imshow_save.show_imgs(images_to_save, wait_ms=1)
            # self.save_images(self._imshow_save.update_whole_image(), file_name)
            # self._imshow_proc.display(1)
            counts = ', '.join(f'merge{n}={len(result_jsons[n])}' for n in range(self.num_merges + 1))
            pbar.write(f'[{i+1}/{len(file_list)}] {os.path.basename(file_name)} instance counts: {counts}')

        self._save_result_jsons(result_jsons)

    def get_linestrings_for_image(self, file_name: str):
        """Extract linestrings and image shape for a single validation image file."""
        image, pred_img, anno_img = self._read_image(file_name)
        self._img_shape = image.shape[:2]
        self._id_count = self.id_offset
        lines, _ = self.extract_lines(pred_img, file_name)
        return lines, self._img_shape

    def _read_image(self, img_file: str):
        image = cv2.imread(img_file)
        # print('image file', img_file)
        pred_file = img_file.replace(self._data_path, self._pred_path).replace('/images/validation/', '/prediction/')
        pred_img = cv2.imread(pred_file)
        anno_img = None
        if self._visualize:
            anno_file = img_file.replace('/images/', '/color_annotations/')
            anno_img = cv2.imread(anno_file)
        # images = {'image': image, 'GT_img': anno_img, 'pred_img': pred_img}
        # self._imshow_base.show_imgs(images)
        return image, pred_img, anno_img
    
    def extract_lines(self, pred_img: np.ndarray, file_name=None) -> Tuple[List[LineString], np.ndarray]:
        line_string_list = []

        debug_img = np.full_like(pred_img, 255)
        for class_id, color in enumerate(self._palette):
            if class_id in self._exclude_classes:
                continue
            # for class_id in [1, 2, 4, 5, 7, 8, 9]:
            pred_class_map = np.all(pred_img == color, axis=-1).astype(np.uint8)
            line_map, line_strings = self._thin_image(pred_class_map, class_id)
            ext_lines = self._extend_lines(line_map, line_strings)
            line_string_list.extend(ext_lines)
            file_name = os.path.basename(file_name)

        line_img = None
        if self._visualize:
            line_img = np.zeros_like(pred_img)
            line_img = self._draw_colored_lines(line_img, line_string_list)
        # self._imshow_proc.show(line_img, 'extracted lines')
        return line_string_list, line_img
    
    def merge_lines(self, src_line_strings: List[LineString], iter: int) -> Tuple[List[LineString], np.ndarray]:
        dst_line_strings = []
        # print(f'=========== [merge_lines] iter={iter}, src_line_strings: {len(src_line_strings)}')
        for class_id, color in enumerate(self._palette):
            if class_id in self._exclude_classes:
                continue
            class_line_strings = [line for line in src_line_strings if line.class_id == class_id]
            merged_lines = self._merge_lines_by_class(class_line_strings, iter_count=1)
            # print(f'[merge_lines] class_id={class_id}, src lines={len(class_line_strings)}, merged={len(merged_lines)}')
            dst_line_strings.extend(merged_lines)

        line_img = None
        if self._visualize:
            line_img = np.zeros([self._img_shape[0], self._img_shape[1], 3], dtype=np.uint8)
            line_img = self._draw_colored_lines(line_img, dst_line_strings)
        # self._imshow_proc.show(line_img, f'merged_lines_{iter}')
        return dst_line_strings, line_img

    def _thin_image(self, seg_map: np.ndarray, class_id: int):
        # print(f'----- [thin_image] -----')
        line_strings = []
        line_map = np.zeros_like(seg_map, dtype=np.int32)
        line_blobs = np.zeros_like(seg_map, dtype=np.int32)
        y, x = np.nonzero(seg_map)
        fill_value = self.id_offset

        show_blobs = line_blobs.copy()

        for k, (y, x) in enumerate(zip(y, x)):
            if line_blobs[y, x] > 0:
                continue

            # floodFill를 통해 seed가 포함된 blob을 채움
            temp = seg_map.copy()
            mask = np.zeros((seg_map.shape[0] + 2, seg_map.shape[1] + 2), np.uint8)
            cv2.floodFill(temp, mask, (x, y), fill_value)
            # 채워진 영역을 바이너리 마스크로 변환 (0 또는 255)
            line_blobs[temp == fill_value] = fill_value
            blob_mask = (temp == fill_value).astype(np.uint8) * 255

            show_blobs = line_blobs.astype(np.int16)

            # cv2.ximgproc.thinning 적용 (얇은 선 추출)
            # (cv2.ximgproc.thinning은 입력이 binary 이미지여야 함)
            line_img = cv2.ximgproc.thinning(blob_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

            # 결과를 line_map에 누적 (겹치는 영역은 덮어쓰기)
            line_map[line_img > 0] = fill_value
            line_strings.append(LineString(id=fill_value, class_id=class_id, peak=(x, y)))
            fill_value += 1

        return line_map.astype(np.uint8), line_strings

    def _extend_lines(self, line_map: np.ndarray, line_strings: List[LineString]) -> List[LineString]:
        # print(f'----- [extend_lines] -----')
        id_list = np.unique(line_map)
        id_list = id_list[id_list >= self.id_offset]
        for line_string in line_strings:
            # 해당 라벨만 추출한 바이너리 이미지
            line_img = (line_map == line_string.id).astype(np.uint8)
            line_string.points = self._sample_points(line_img, self.sample_stride)
            if line_string.points.shape[0] < 2:
                line_string.id = None
                continue
            line_string.length = np.sum(np.linalg.norm(np.diff(line_string.points, axis=0), axis=1))
            if line_string.length < 3:
                line_string.id = None
                continue
            line_string = self._extrapolate_line(line_string, self.extend_len, self.sample_stride)

        line_strings = [ls for ls in line_strings if ls.id is not None]
        # 선 길이에 따라 내림차순 정렬
        line_strings.sort(key=lambda ls: ls.length, reverse=True)
        return line_strings

    def _sample_points(self, line_img: np.ndarray, stride: int) -> np.ndarray:
        rows, cols = np.nonzero(line_img)
        points = np.stack((cols, rows), axis=1)
        if points.shape[0] < 2:
            return points
        sorted_points = [points[0]]
        direction = points[1] - points[0]
        sorted_points = self._sort_to_direction(points, sorted_points, True, direction, stride)
        sorted_points = self._sort_to_direction(points, sorted_points, False, -direction, stride)
        sorted_points = np.array(sorted_points).astype(np.int32)
        return sorted_points.astype(np.int32)

    def _sort_to_direction(self, src_points: np.ndarray, sorted_points: List[np.ndarray], to_tail: bool,
                           direction: np.ndarray, stride: int) -> List[np.ndarray]:
        points = src_points.copy()
        while len(points) > 0:
            last_point = sorted_points[-1] if to_tail else sorted_points[0]
            vecs = points - last_point
            distances = np.sqrt(np.sum(vecs ** 2, axis=1))
            dir_norm = float(np.linalg.norm(direction))
            # 진행 방향과 후보 스텝 사이 각의 코사인
            with np.errstate(invalid='ignore', divide='ignore'):
                cos_ang = np.sum(vecs * direction, axis=1) / (distances * dir_norm + 1e-9)
            # 유효 후보: 거리 [stride,30) & 전방(90° cone). 하드 게이트로 선을 끊지 않는다.
            valid_mask = (distances < 30) & (distances >= stride) & (cos_ang >= 0)
            if np.sum(valid_mask) == 0:
                break
            # 거리 + 곡률 패널티로 선택 → 분기에서 곧게 이어지는 후보를 선호(직선은 그대로 통과)
            score = distances * (1.0 + self.turn_penalty * (1.0 - cos_ang))
            score[~valid_mask] = np.inf
            next_index = np.argmin(score)
            if to_tail:
                sorted_points.append(points[next_index])
                # print(f'[sort_to_direction] next point: {sorted_points[-1]}')
                direction = sorted_points[-1] - last_point
            else:
                sorted_points.insert(0, points[next_index])
                # print(f'[sort_to_direction] next point: {sorted_points[0]}')
                direction = sorted_points[0] - last_point
            distances = np.sqrt(np.sum((points - last_point) ** 2, axis=1))
            points = points[distances >= stride]
        return sorted_points

    def _extrapolate_line(self, line_string: LineString, extend_len: int, stride: int) -> LineString:
        points = line_string.points  # (N,2) 배열
        N = len(points)
        n_ext = extend_len // stride

        head_prev_idx = min(max(N // 3, 3), N - 1)
        head_ext = self._extend_endpoint(points[0], points[head_prev_idx], n_ext, stride)
        tail_prev_idx = N - 1 - min(max(N // 3, 3), N - 1)
        tail_ext = self._extend_endpoint(points[-1], points[tail_prev_idx], n_ext, stride)

        n_head = len(head_ext)
        ext_points = np.vstack([head_ext[::-1], points, tail_ext]) if n_head > 0 else np.vstack([points, tail_ext])
        ext_points = np.clip(ext_points, [0, 0], [self._img_shape[1], self._img_shape[0]])

        line_string.ext_points = np.rint(ext_points).astype(np.int32)
        line_string.src_range = (n_head, n_head + N - 1)
        return line_string

    def _extend_endpoint(self, tip: np.ndarray, prev: np.ndarray, n_ext: int, stride: int) -> np.ndarray:
        direction = tip - prev
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.empty((0, 2), dtype=np.float64)
        direction = direction / norm
        return np.array([tip + direction * stride * i for i in range(1, n_ext + 1)])

    def _clean_lines(self, lines: List[LineString]) -> List[LineString]:
        """추출된 선들을 공유 모듈로 정리한다(merge_annotation과 동일 기조).

        - 클래스별 복제선은 가장 긴 대표만 남김(dedup_keep_longest) → 점 섞임 없음.
        - center_line의 평행 겹침은 가장 긴 기준선을 두고 짧은 선의 겹침 구간만 잘라냄
          (trim_overlaps, hysteresis) → 갈라지는 가지는 별도 선으로 보존.
        점을 평균/재정렬하지 않으므로 지그재그가 생기지 않는다."""
        by_cls = {}
        for l in lines:
            if l.points is not None and len(l.points) >= 2:
                by_cls.setdefault(l.class_id, []).append(l)

        cleaned = []
        for cid, group in by_cls.items():
            kept = [group[k] for k in pm.dedup_keep_longest(
                [g.points for g in group], self.dup_dist, self.dup_ratio)]
            if cid == self.trim_class_id:
                for src_i, pts in pm.trim_overlaps(
                        [k.points for k in kept],
                        overlap_high=self.overlap_dist, overlap_low=self.overlap_low,
                        min_free_len=self.min_free_len, bridge_gap=self.bridge_gap, step=self.trim_step):
                    base = kept[src_i]
                    cleaned.append(LineString(id=base.id, peak=base.peak, class_id=cid,
                                              points=np.rint(pts).astype(np.int32),
                                              length=pm.arc_length(pts)))
            else:
                cleaned.extend(kept)

        # id 고유성 보장(라벨맵 충돌 방지) + 변경된 점들의 확장점 재계산
        out = []
        for i, l in enumerate(cleaned):
            if l.points is None or len(l.points) < 2:
                continue
            l.id = self.id_offset + i
            out.append(self._extrapolate_line(l, self.extend_len, self.sample_stride))
        return out

    def _residual_pred(self, pred_img: np.ndarray, lines: List[LineString]) -> np.ndarray:
        """1차 추출 선들의 자취(두께 residual_remove_width)를 seg map에서 지운 잔여 예측 이미지.
        지워진 픽셀은 배경(0)으로 만들어 잔여 추출 시 같은 선이 다시 잡히지 않게 한다."""
        mask = np.zeros((self._img_shape[0], self._img_shape[1]), dtype=np.uint8)
        for l in lines:
            if l.points is None or len(l.points) < 2:
                continue
            pts = l.points.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=self.residual_remove_width)
        residual = pred_img.copy()
        residual[mask > 0] = 0
        return residual

    def _combine_dedup(self, lines: List[LineString]) -> List[LineString]:
        """1차+잔여 선을 합친 뒤 클래스별 복제만 제거하고 id를 고유화한다(trim 재적용 안 함).
        잔여 추출에서 1차 선과 거의 겹친 찌꺼기 선이 잡히면 여기서 제거된다."""
        by_cls = {}
        for l in lines:
            if l.points is not None and len(l.points) >= 2:
                by_cls.setdefault(l.class_id, []).append(l)
        kept = []
        for group in by_cls.values():
            kept += [group[k] for k in pm.dedup_keep_longest(
                [g.points for g in group], self.dup_dist, self.dup_ratio)]
        out = []
        for i, l in enumerate(kept):
            l.id = self.id_offset + i
            out.append(self._extrapolate_line(l, self.extend_len, self.sample_stride))
        return out

    def _smoothed_copies(self, lines: List[LineString]) -> List[LineString]:
        """점들을 이동평균으로 스무딩한 복사본을 만든다(원본 lines는 다음 병합용으로 보존).
        thinning/병합 경계에서 생기는 지그재그를 완화해 선을 반듯하게 편다."""
        out = []
        for l in lines:
            if l.points is None or len(l.points) < 3:
                out.append(l)
                continue
            sp = pm.smooth_polyline(l.points, self.smooth_window, self.smooth_iters)
            out.append(LineString(id=l.id, peak=l.peak, class_id=l.class_id,
                                  points=np.rint(sp).astype(np.int32), length=l.length))
        return out

    def _merge_lines_by_class(self, line_strings: List[LineString], iter_count: int = 0) -> List[LineString]:
        """끝-끝으로 이어지는 선들을 직렬 연결로 병합한다(점 단위 NN 재정렬 안 함).

        후보쌍은 기존처럼 확장 끝선분 겹침(_find_overlap)으로 찾되, 평행 본체 쌍은
        거부(bodies_parallel)해 평행 가닥 오병합으로 인한 지그재그를 막는다.
        각 그룹은 폴리라인 단위 체이닝(concat_polylines_in_series)으로 이어 붙여
        각 선의 내부 점 순서를 보존한다."""
        lines = [l for l in line_strings
                 if l.id is not None and l.points is not None and len(l.points) >= 2]
        n = len(lines)
        if n == 0:
            return []
        id2idx = {l.id: i for i, l in enumerate(lines)}
        parent = list(range(n))
        members = {i: [i] for i in range(n)}
        find = pm.make_find(parent)
        par_cache = {}

        def parallel(i, j):
            key = (i, j) if i < j else (j, i)
            if key not in par_cache:
                par_cache[key] = pm.bodies_parallel(
                    lines[i].points, lines[j].points, self.parallel_overlap, self.parallel_lateral)
            return par_cache[key]

        # 후보쌍: 확장 끝선분이 겹치는 같은 클래스 쌍
        candidates = set()
        for i, line in enumerate(lines):
            for oid in self._find_overlap(lines, line):
                j = id2idx.get(int(oid))
                if j is not None and j != i:
                    candidates.add((i, j) if i < j else (j, i))

        # 그룹 인식 union: 평행 가닥은 병합 거부 (전이적 평행 오병합 방지)
        for i, j in candidates:
            if parallel(i, j):
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            mi, mj = members[ri], members[rj]
            if any(parallel(p, q) for p in mi for q in mj):
                continue
            parent[rj] = ri
            members[ri] = mi + mj
            members.pop(rj, None)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        out = []
        for idxs in groups.values():
            if len(idxs) == 1:
                out.append(lines[idxs[0]])
                continue
            base = lines[idxs[0]]
            chained = pm.concat_polylines_in_series([lines[k].points for k in idxs])
            base.points = np.rint(chained).astype(np.int32)
            base.length = pm.arc_length(base.points)
            out.append(self._extrapolate_line(base, self.extend_len, self.sample_stride))
        return out

    def _find_overlap(self, line_strings: List[LineString], this_line: LineString) -> Set[int]:
        src_line_id = this_line.id
        if this_line.ext_points is None:
            return set()
        line_strings_copy = line_strings.copy()
        line_strings_copy.remove(this_line)
        dilated_map = self._draw_line_strings(line_strings_copy, extend=True)
        dilated_map = cv2.dilate(dilated_map, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8))
        label_map = dilated_map.copy() if dilated_map.ndim == 2 else dilated_map[:, :, 0].copy()

        line_img = np.zeros_like(dilated_map, dtype=np.uint8)
        pts = this_line.ext_points.reshape((-1, 1, 2))
        cv2.polylines(line_img, [pts], isClosed=False, color=(255, 255, 255), thickness=3)
        line_map = line_img[:, :, 0]

        label_map[line_map == 0] = 0
        overlap_labels = label_map[label_map > 0]
        if overlap_labels.size == 0:
            return set()
        unique, counts = np.unique(overlap_labels, return_counts=True)

        mask_cond = (unique != 0) & (unique != src_line_id) & (counts > self.overlap_thresh)
        label_ids = set(unique[mask_cond].tolist())
        # print(f'unique: {unique}, counts: {counts}, label_ids: {label_ids}')
        return label_ids

    def _save_result_jsons(self, result_jsons: list):
        names = ['origin'] + [f'merge{n}' for n in range(1, self.num_merges + 1)]
        for name, data in zip(names, result_jsons):
            path = os.path.join(self._result_path, f'coco_pred_instances_{name}.json')
            with open(path, 'w') as f:
                json.dump(data, f)
        counts = ', '.join(f'{name}={len(data)}' for name, data in zip(names, result_jsons))
        print(f'FINAL instance counts: {counts}\n')

    def _exclude_short_lines(self, line_strings: List[LineString]) -> Tuple[List[LineString], np.ndarray]:
        filtered_line_strings = []

        # [디버깅] 입력된 라인 개수 출력
        print(f'>> [DEBUG] _exclude_short_lines input count: {len(line_strings)}')

        rejected_count = 0
        min_len = float('inf')
        max_len = 0

        for line in line_strings:
            # 기존 로직: 길이 재계산
            line.length = np.sum(np.linalg.norm(np.diff(line.points, axis=0), axis=1))

            # [디버깅] 최소/최대 길이 추적
            if line.length < min_len: min_len = line.length
            if line.length > max_len: max_len = line.length

            if line.length > self.short_length:
                filtered_line_strings.append(line)
            else:
                # [디버깅] 탈락하는 라인의 정보 출력
                rejected_count += 1
                # print(f'   >> Rejected Line ID: {line.id}, Length: {line.length:.2f}')

        # [디버깅] 결과 요약 출력
        print(f'>> [DEBUG] Threshold: {self.short_length}')
        print(f'>> [DEBUG] Length Range: Min={min_len:.2f}, Max={max_len:.2f}')
        print(f'>> [DEBUG] Rejected: {rejected_count}, Survived: {len(filtered_line_strings)}')

        line_img = np.zeros([self._img_shape[0], self._img_shape[1], 3], dtype=np.uint8)
        line_img = self._draw_colored_lines(line_img, filtered_line_strings)
        return filtered_line_strings, line_img

    def _draw_line_strings(self, line_strings: List[LineString], extend=False, color=None):
        image = np.zeros((self._img_shape[0], self._img_shape[1], 3), dtype=np.uint8)
        for line in line_strings:
            if line.id is None or line.points is None:
                continue
            if extend:
                pts = line.ext_points.reshape((-1, 1, 2))
            else:
                pts = line.points.reshape((-1, 1, 2))
            line_color = (line.id, line.id, line.id) if color is None else color
            cv2.polylines(image, [pts], isClosed=False, color=line_color, thickness=3)
        return image

    def _draw_single_line(self, line_string : LineString, thickness=None, extend=False):
        image = np.zeros((self._img_shape[0], self._img_shape[1], 3), dtype=np.uint8)
        if line_string.id is None or line_string.points is None:
            return image
        pts = line_string.points.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=(line_string.id, line_string.id, line_string.id), thickness=thickness)
        return image

    def _draw_colored_lines(self, pred_img, line_strings : List[LineString], extended=False, select=None):
        image = pred_img.copy()
        for line in line_strings:
            if line.id is None:
                continue
            color = self._palette[line.class_id]
            if extended:
                pts = line.ext_points.reshape((-1, 1, 2))
            else:
                pts = line.points.reshape((-1, 1, 2))
            if select is None:
                cv2.polylines(image, [pts], isClosed=False, color=color, thickness=3)
            elif select is not None:
                if line.class_id == select:
                    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=3)
                else:
                    continue
        return image

    def _draw_blobs_with_color(self, line_map):
        n_labels = int(np.max(line_map))
        rng = np.random.default_rng(42)  # 재현성을 위해 시드 고정(원하면 제거)
        H = rng.uniform(0, 180, size=n_labels + 1)  # [0,180)
        S = rng.uniform(170, 255, size=n_labels + 1)  # 채도 ↑ (170~255)
        V = rng.uniform(130, 220, size=n_labels + 1)  # 명도 ↓ (130~220)
        H[0], S[0], V[0] = 0, 0, 0
        hsv = np.stack([H, S, V], axis=1).astype(np.uint8).reshape(-1, 1, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)  # shape: (n_labels+1, 3)
        colorized = bgr[line_map]
        return colorized

    def convert_to_json(self, line_strings: List[LineString], image_id: str):
        pred_json = []
        for line in line_strings:
            mask = self._draw_single_line(line, self.thickness)
            mask = (np.all(mask > 0, axis=-1)).astype(np.uint8)
            mask = np.asfortranarray(mask)
            rle = maskUtils.encode(mask)
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')
            prediction = {
                "image_id": image_id,
                "category_id": line.class_id,
                "segmentation": rle,
                "score": 1.,
            }
            pred_json.append(prediction)
        return pred_json

    def save_images(self, save_image, img_file):
        filename = img_file.replace('/images/validation', '/result/results')
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        print('save filename:', filename)
        cv2.imwrite(filename, save_image)


def main():
    from util import find_best_pred_json_path
    csv_path = os.path.join(cfg.RESULT_PATH, 'total_performance.csv')
    model_name, _, _ = find_best_pred_json_path(csv_path)
    
    if model_name is None:
        model_name = "internimage_large"
        
    model_dir = cfg.MODEL_PREFIX + model_name
    model_type = "Internimage" if "internimage" in model_name.lower() else "mask2former"
    model_path = os.path.join(cfg.DATA_ROOT, model_type, model_dir)
    
    line_detector = LineStringDetector(cfg.DATASET_PATH, model_path, cfg.RESULT_PATH)
    line_detector.detect_lines()


if __name__ == '__main__':
    main()

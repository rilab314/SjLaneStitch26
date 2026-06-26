"""Figure 6 — 병합: 외삽 교차 + 평행 거부 + 직렬 체이닝 (1×3 가로 콜라주).

패널: (a) 정제 단편 본체(끊긴 상태가 보이도록) | (b) 본체 + 끝점 바깥쪽 외삽(연회색)
      | (c) 직렬 연결된 최종 결과.
단편이 실제로 이어지고(joined ≥ 3) center_line 평행 거부가 일어나는 프레임만 출력한다.
"""
import os
import sys

_CUR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_CUR, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import figure_render as fr
import figure_metrics as fm
from figure_base import FigureGenerator
from lane_stitcher import bodies_parallel


class MergingFigure(FigureGenerator):
    """merge 전 단편 | 외삽 | merge 후를 1×3으로 보인다(끝–끝 체이닝과 평행 오병합 방지)."""

    name = "Figure_6"
    cls = fm.CENTER_LINE_ID
    ext_color = (153, 136, 119)   # LightSlateGray(BGR) — 연한 청회색, 어떤 클래스 렌더색과도 안 겹침
    min_joined = 3

    def build_figure(self, image_id, path):
        stage = self._detector.stage_linestrings(
            path, do_merge=True, merge_iters=self._detector.num_merges)
        if fm.measure_merge(stage)["joined"] < self.min_joined:
            return None
        self._detector._img_shape = stage["img_shape"]
        if self.count_parallel_rejections(stage["refined"]) < 1:
            return None
        return self.compose(stage), ""

    def count_parallel_rejections(self, refined):
        """병합 시 평행 이중선으로 거부될 center_line 쌍 수(figure 6의 핵심 장면)."""
        group = self.center_candidates(refined)
        if len(group) < 2:
            return 0
        by_id = {strand.id: strand for strand in group}
        return sum(self._rejections_from(strand, group, by_id) for strand in group)

    def center_candidates(self, refined):
        """확장 끝선분이 있는 center_line 후보 리스트."""
        return [s for s in refined if s.class_id == self.cls
                and s.points is not None and len(s.points) >= 2 and s.ext_points is not None]

    def _rejections_from(self, strand, group, by_id):
        """strand와 겹치는 후보 중 본체가 평행한 쌍 수(중복 방지로 id 큰 쪽만 카운트)."""
        detector = self._detector
        count = 0
        for other_id in detector._find_overlap(group, strand):
            other = by_id.get(int(other_id))
            if other is None or other.id <= strand.id:
                continue
            if bodies_parallel(strand.points, other.points,
                               detector.parallel_overlap, detector.parallel_lateral):
                count += 1
        return count

    def compose(self, stage):
        """merge 전 | 외삽 | merge 후 세 패널을 검은 여백으로 가로 결합한다."""
        height, width = stage["img_shape"]
        refined = stage["refined"]
        final = self._detector._smoothed_copies(self.final_merge(stage))
        panels = [
            fr.draw_strands(fr.make_white_canvas(height, width), refined, dots=True),
            self.extension_panel(refined, height, width),
            fr.draw_strands(fr.make_white_canvas(height, width), final, dots=True),
        ]
        return fr.concat_horizontal(panels)

    def extension_panel(self, refined, height, width):
        """(b) 단편 본체(클래스색 + 끝점 점) 위에 끝점 바깥쪽 외삽만 연회색선으로."""
        canvas = fr.draw_strands(fr.make_white_canvas(height, width), refined, dots=True)
        for strand in refined:
            fr.draw_extension(canvas, strand.ext_points, strand.src_range, self.ext_color)
        return canvas


if __name__ == "__main__":
    MergingFigure().run()

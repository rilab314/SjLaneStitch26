# CLAUDE.md

이 파일은 Claude Code(claude.ai/code)가 이 저장소에서 작업할 때 참고하는 안내 문서입니다.

## 언어 설정

**모든 답변과 설정 파일은 한글로 작성한다.**

---

## 프로젝트: LaneStitch

**LaneStitch** — 위성영상 세그멘테이션을 차선 폴리라인으로 벡터화하고, 단편화된 선을
이어 붙여(stitch) 통합 차선을 만드는 파이프라인. 다음 순서로 동작한다:
1. InternImage/Mask2Former 세그멘테이션 모델의 예측 결과와 ADE20K 형식의 GT를 입력으로 받음
2. 세그멘테이션 블롭을 Zhang-Suen 세선화(thinning)를 통해 `Strand`(폴리라인) 인스턴스로 변환
3. 끝점 겹침 감지를 통해 단편화된 선을 병합(stitch)
4. 객체 단위 F1(그리디 IoU 매칭, IoU 0.5)과 픽셀 단위 mIoU로 성능 평가
5. 논문 작성을 위한 CSV 테이블 및 시각화 그림(Figure) 생성

### 용어 정리 (line/lane 혼동 방지)
- **lane**: 의미 차선 객체(center_line 등 클래스) — 도메인 관점
- **polyline / `Strand`**: 그 차선의 기하 표현(순서 있는 점열) — 자료구조 관점.
  세그멘테이션에서 추출돼 stitch될 "가닥". `LaneStitcher`가 다루는 단위.
- **`LaneStitcher`**: 세그멘테이션 → lane 벡터화·병합 파이프라인 클래스 (`core/lane_stitcher.py`)
- **`MergeAnnotator`**: GT 어노테이션 통합 파이프라인 (`dataprep/merge_annotation.py`)
- 공유 기하/병합 연산은 각 파이프라인 파일(`core/lane_stitcher.py`, `dataprep/merge_annotation.py`) 안에 포함되어 있다

---

## 실행 방법

`src/`는 종류별 하위 폴더로 정리되어 있다. 직접 실행하는 스크립트 목록·순서의 상세는
`src/README.md`(영문)를 참고한다. 모든 명령은 `src/`에서 실행한다.

폴더 구조 요약:
- `core/` — 공유 라이브러리(직접 실행 X): `lane_stitcher.py`, `evaluator.py`,
  `stitch_config.py`, `util.py`, `show_imgs.py`
  (`evaluator.py`·`lane_stitcher.py`의 `main()`은 실제 실행용이 아니라 동작 확인용 스모크 테스트다)
- `dataprep/` — 데이터셋 빌드: `build_dataset.py`(메인, SEED 원본 → `ade20k/`+`coco/` 전체 split),
  `make_seg_labels.py`(ADE20K 인덱스 라벨 생성 라이브러리/부분 재생성), `merge_annotation.py`(COCO 병합 GT 라이브러리/부분 재생성)
- `inference/` — 세그멘테이션 추론 → 모델별 `pred_val`/`pred_test` (입력 이미지는 `ade20k/images/<split>`)
- `experiment/` — 파이프라인 실행·평가 (`run_experiments.py`, `run_parallel_sweep.py`(병렬 재현),
  `run_best_experiment.py`)
- `tables/` — 논문 Table 1~5 (`table_1.py`~`table_5.py`, `num_params.py`, 공통 헬퍼 `table_common.py`)
- `figures/` — 논문 Figure 1~8 (`figure_1.py`~`figure_8.py`, 공통 렌더 헬퍼 `figure_*.py`)
- `import` 경로는 `_bootstrap.py`가 `core/`·`tables/`·`figures/`를 sys.path에 등록해 유지된다.

데이터 중복 없음: 모든 GT(인덱스 라벨·컬러 라벨·COCO 병합 GT)는 `ade20k/`·`coco/`에 한 벌만 존재하고,
모든 스크립트가 `config`의 `image_dir/label_dir/color_label_dir/coco_anno_path` 헬퍼로 그 한 벌을 참조한다.
원본 SEED(`satellite_good_matching_250206`)는 `build_dataset.py`만 읽는다.

```bash
cd src

# 1. 데이터셋 빌드 (SEED 원본 → ade20k/ + coco/, 최초 1회. train/val/test 전체)
python dataprep/build_dataset.py                     # 옵션: --split validation test / --skip images

# 2. 세그멘테이션 추론 → <model>/pred_val, <model>/pred_test (ade20k/images 입력)
python inference/infer_internimage.py
python inference/infer_mask2former.py

# 3. 파이프라인 실행·평가 (validation에서 파라미터 탐색 후 test에 best 적용)
python experiment/run_experiments.py --split validation test
MAXJOBS=14 python experiment/run_parallel_sweep.py   # 동일 결과 병렬 재현(권장, 훨씬 빠름)
python experiment/run_experiments.py --fast          # 창·콜라주 생략 고속 평가
python experiment/run_experiments.py --eval-only     # 예측 재사용, 평가만 재실행(GT 변경 시)
python experiment/run_best_experiment.py             # best 조합 단일 실행

# 4. 테이블 (→ RESULT_PATH/Tables/*.csv)
python tables/num_params.py              # 모델별 파라미터 수 (Table 1 Params 열)
python tables/table_1.py                 # 모델 비교 (segmentation vs merge×1), val·test
python tables/table_2.py                 # best 모델 클래스별 성능
python tables/table_3.py                 # best 모델 클래스별 진단 분해
python tables/table_4.py                 # 단계별 향상 (baseline→residual→refinement→merge1→merge2)
python tables/table_5.py                 # 파라미터 ablation (stride·extend·turn)

# 5. 그림 (→ RESULT_PATH/Figure/*)
python figures/figure_1.py               # ... figure_8.py 까지
```

---

## 설정

`src/config.py`에 모든 경로가 정의되어 있다. **실행 전 반드시 수정해야 한다:**
- `DATA_ROOT`: 데이터셋과 모델 출력이 저장된 루트 디렉토리
- `SRC_DATASET_PATH`: 원본 SEED 소스 (`image/`·`label/`·`dataset.json`). `build_dataset.py`만 읽는다
- `DATASET_PATH`: 빌드된 ADE20K 데이터셋 경로 (`DATA_ROOT/ade20k`) — 이미지·인덱스 라벨·컬러 라벨
- `COCO_PATH`: 빌드된 COCO 데이터셋 경로 (`DATA_ROOT/coco`) — 병합 인스턴스 GT + 이미지
- `RESULT_DIR`: 출력(예측 JSON·CSV·Figure·Table) 저장 폴더명. 실행마다 바꿔 결과를 분리 보관한다 (예: `results_260709`)
- `RESULT_PATH`: 출력 경로 (`DATA_ROOT/RESULT_DIR`)
- `EVAL_SPLITS`: 평가 대상 split (`validation`, `test`) / `ALL_SPLITS`: 빌더가 만드는 전체 split

split→경로 헬퍼로 모든 스크립트가 데이터를 참조한다(중복 없음):
- `image_dir(split)` = `ade20k/images/<split>`, `label_dir(split)` = `ade20k/annotations/<split>`(mIoU GT),
  `color_label_dir(split)` = `ade20k/color_annotations/<split>`
- `coco_anno_path(split)` = `coco/annotations/instances_<split>2017.json`(객체 F1 GT),
  `coco_image_dir(split)` = `coco/<split>2017`
- `LABEL_PATH`/`COCO_MERGED_ANNO_PATH`/`COCO_ANNO_PATH`/`DATA_PATH`는 validation 기본 별칭(기존 스크립트 호환)
- ADE20K는 train split 폴더명이 `training`이다(`ADE_SPLIT_DIR`), COCO 이미지 폴더는 `train2017/val2017/test2017`(`COCO_IMG_DIR`)

### 클래스 메타정보
- `METAINFO`: 클래스 ID, 이름, RGB 색상을 정의하는 딕셔너리 리스트. ID 0은 ignore, ID 1~11이 실제 차선 클래스
- `EXCLUDE_IDS`: 평가에서 제외할 클래스 ID 목록 (`[0, 8, 10]`, bicycle_lane(11)은 평가 포함)
- `EVAL_CLASS_IDS`: 실제 평가에 사용되는 클래스 ID 목록 (METAINFO에서 EXCLUDE_IDS를 제외하고 자동 생성)
- `ID2BGR`: 클래스 ID → BGR 색상 튜플 매핑. 예측 이미지에서 클래스별 픽셀을 추출할 때 사용
- `ID2NAME`: 클래스 ID → 클래스 이름 문자열 매핑. 테이블 출력 시 사용

### 시각화 전용 메타정보
- `RENDER_METAINFO`: 논문 그림 시각화 렌더링에 최적화된 클래스별 RGB 색상 설정 (예: `stop_line`(ID 9)은 명확한 구분을 위해 금색 `(255, 215, 0)`으로 지정)
- `RENDER_ID2BGR`: `RENDER_METAINFO`를 기반으로 자동 생성되는 BGR 색상 매핑 딕셔너리

### 기타
- `MODEL_PREFIX`: 모델 디렉토리 이름의 공통 접두사 (`"satellite_ade20k_250925_"`). `run_experiments.py`에서 CSV에 모델명 저장 시 이 접두사를 제거하는 데 사용

새로운 설정 작성 시 `src/config-template.py`를 참고한다.

---

## 아키텍처

### 데이터 흐름

```
satellite_good_matching_250206/{image,label}   ← 원본 SEED 소스
        ↓ dataprep/build_dataset.py
ade20k/images/<split>/*.png                      ← 입력 위성 이미지
ade20k/annotations/<split>/*.png                 ← GT 인덱스 레이블 (mIoU)
coco/annotations/instances_<split>2017.json      ← GT 병합 인스턴스 (객체 F1)
Internimage/ (또는 mask2former/)
  └─ <model_name>/{pred_val,pred_test}/*.png     ← 세그멘테이션 모델 출력 (클래스별 색상 코딩)
        ↓
LaneStitcher.detect_lines()
        ↓
coco_pred_{val,test}_origin.json                 # 초기 벡터화 결과 (skeletonization 직후)
coco_pred_{val,test}_merge{1,2}.json             # 단계별 병합 결과 (merge2가 최종)
        ↓
[Table 생성]  (논문 Table 1~5, 공통 헬퍼 tables/table_common.py)
tables/num_params.py → num_params.csv             # 모델별 파라미터 수
tables/table_1.py    → table_1.csv                # 모델 비교 (segmentation vs merge×1), val·test
tables/table_2.py    → table_2.csv                # best 모델 클래스별 성능 (count·F1@0.5·mIoU)
tables/table_3.py    → table_3.csv                # best 모델 클래스별 진단 분해 (6지표)
tables/table_4.py    → table_4.csv                # 단계별 향상 (baseline→residual→refinement→merge1→merge2)
tables/table_5.py    → table_5.csv                # 파라미터 ablation (stride·extend·turn)
        ↓
[Figure 생성]
figures/figure_1.py ~ figures/figure_8.py → RESULT_PATH/Figure/*   # 논문 Figure 1~8 콜라주
```

## 런타임 디렉토리 구조

```
DATA_ROOT/
  satellite_good_matching_250206/          # 원본 SEED 소스 (build_dataset.py만 읽음)
    ├─ image/*.png                         # 위성 이미지 (train+val+test = 12828)
    ├─ label/*.json                        # SEED 벡터 라벨
    └─ dataset.json                        # split별 basename 목록
  ade20k/                                   # 빌드된 ADE20K 시맨틱 세그 데이터셋
    images/{training,validation,test}/*.png            # 위성 이미지
    annotations/{training,validation,test}/*.png       # 인덱스 라벨 (pixel = class_id+1, mIoU GT)
    color_annotations/{training,validation,test}/*.png # 컬러 시각화 라벨
  coco/                                     # 빌드된 COCO 인스턴스 세그 데이터셋
    ├─ annotations/instances_{train,validation,test}2017.json   # 병합 인스턴스 GT (객체 F1)
    ├─ annotations/instances_{...}2017_selected.json            # 평가 캐시(EXCLUDE_IDS 필터+id/area/iscrowd, 자동 생성/무효화)
    ├─ {train2017,val2017,test2017}/*.png                       # 위성 이미지
    └─ class_counts.csv                                         # split×클래스 인스턴스 수
  Internimage/
    ├─ checkpoint/*.pth                     # InternImage 체크포인트
    └─ satellite_ade20k_250925_internimage_large/
         ├─ pred_val/*.png  pred_test/*.png # 색상 코딩된 세그멘테이션 예측
         └─ metrics_{validation,test}.json  # segmentation-prediction mIoU 캐시
  mask2former/
    ├─ checkpoint/  pre_trained/            # Mask2Former 체크포인트
    └─ satellite_ade20k_250925_mask2former_{large,small}/{pred_val,pred_test}/*.png
  results_<date>/                           # RESULT_DIR (실행마다 분리 보관)
    ├─ total_performance.csv                # 하이퍼파라미터 전체 성능 탐색 결과
    ├─ num_params.csv                       # 모델별 파라미터 수
    ├─ _combo_logs/                         # (run_parallel_sweep) 조합별 로그
    ├─ <model_name>/
    │    └─ thick=T,stride=S,extend=E,turn=P/          # 하이퍼파라미터 조합별 출력
    │         ├─ coco_pred_{val,test}_{origin,merge1,merge2}.json
    │         └─ eval_result.csv
    ├─ Tables/table_1.csv .. table_5.csv
    └─ Figure/Figure_1 .. Figure_8, ...     # figure_1.py ~ figure_8.py 출력
```

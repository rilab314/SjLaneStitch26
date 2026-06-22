# CLAUDE.md

이 파일은 Claude Code(claude.ai/code)가 이 저장소에서 작업할 때 참고하는 안내 문서입니다.

## 언어 설정

**모든 답변과 설정 파일은 한글로 작성한다.**

---

## 프로젝트 개요

위성 이미지 차선 감지 파이프라인으로, 다음 순서로 동작한다:
1. InternImage/Mask2Former 세그멘테이션 모델의 예측 결과와 ADE20K 형식의 GT를 입력으로 받음
2. 세그멘테이션 블롭을 Zhang-Suen 세선화(thinning)를 통해 `LineString` 인스턴스로 변환
3. 끝점 겹침 감지를 통해 단편화된 선분을 병합
4. COCO AP(IoU 임계값 0.10, 0.20, 0.50)와 픽셀 단위 mIoU로 성능 평가
5. 논문 작성을 위한 CSV 테이블 및 시각화 그림(Figure) 생성

---

## 실행 방법

모든 스크립트는 `src/` 디렉토리에서 실행해야 한다 (모듈 임포트가 상대 경로 기준):

```bash
cd src

# 단일 모델 설정으로 차선 감지 실행
python lane_detector.py

# 모든 모델과 파라미터 조합으로 전체 하이퍼파라미터 탐색 실행
python run_experiments.py

# 1. 테이블(Tables) 생성 스크립트 실행
# 체크포인트를 로드하여 모델별 파라미터 개수를 계산하고 num_params.csv 생성
python Table/num_params.py

# 단일 실행 결과 평가 및 table_1.csv 생성
python Table/table_1.py

# 최적 모델/파라미터를 찾고 table_2.csv (클래스별 지표)와 table_3.csv (클래스별 심화 지표) 생성
python Table/table_2.py

# 논문용 Ablation Study 테이블 (table_4.csv) 생성
python Table/table_4.py

# 2. 그림(Figures) 생성 스크립트 실행
# 최적 예측 JSON 결과를 기반으로 validation 이미지에 개별 마스크 시각화 이미지 생성 (Figure_1)
python Figure/figure_1.py

# 원본 위성 이미지와 Figure_1 마스크 시각화를 1x2 쌍으로 결합한 최종 콜라주(figure1.jpg) 생성
python Figure/figure_1_fin.py

# 원본+GT overlay, Segmentation 예측, 초기 LineString, 최종 병합 LineString을 담은 2x2 콜라주 생성 (Figure_2)
python Figure/figure_2.py

# Guiding line(8)과 Safety zone(10)에 대해 GT와 예측 결과를 비교한 콜라주 생성 (Figure_3)
python Figure/figure_3.py

# center_line(1)의 병합 알고리즘 중간 단계를 시각화하는 1x4 콜라주 생성 (Figure_4)
python Figure/figure_4.py

# 원본 이미지, 원본+GT overlay, 원본+Prediction (merge2) overlay를 담은 1x3 콜라주 생성 (Figure_5)
python Figure/figure_5.py

# 원본+GT overlay와 세 모델(internimage_large, mask2former_large, mask2former_small)의
# segmentation overlay를 비교하는 2x2 콜라주 생성 (Figure_compare, 20px 흰색 구분선)
python Figure/figure_compare.py
```

---

## 설정

`src/config.py`에 모든 경로가 정의되어 있다. **실행 전 반드시 수정해야 한다:**
- `DATA_ROOT`: 데이터셋과 모델 출력이 저장된 루트 디렉토리
- `DATASET_PATH`: ADE20K 형식의 데이터셋 경로 (`DATA_ROOT/ade20k`)
- `RESULT_PATH`: JSON 예측 결과와 CSV 테이블이 저장되는 경로
- `COCO_MERGED_ANNO_PATH`: GT COCO 어노테이션 JSON 파일 경로 (`RESULT_PATH/merged_annotations.json`)
- `DATA_PATH`: 데이터셋 경로 (`DATASET_PATH`와 동일)
- `LABEL_PATH`: GT 레이블 이미지 경로 (`DATASET_PATH/annotations/validation`)
- `COCO_ANNO_PATH`: COCO 형식 GT 어노테이션 파일 경로 (`COCO_MERGED_ANNO_PATH`와 동일)

### 클래스 메타정보
- `METAINFO`: 클래스 ID, 이름, RGB 색상을 정의하는 딕셔너리 리스트. ID 0은 ignore, ID 1~11이 실제 차선 클래스
- `EXCLUDE_IDS`: 평가에서 제외할 클래스 ID 목록 (`[0, 8, 10, 11]`)
- `EVAL_CLASS_IDS`: 실제 평가에 사용되는 클래스 ID 목록 (METAINFO에서 EXCLUDE_IDS를 제외하고 자동 생성)
- `ID2BGR`: 클래스 ID → BGR 색상 튜플 매핑. 예측 이미지에서 클래스별 픽셀을 추출할 때 사용
- `ID2NAME`: 클래스 ID → 클래스 이름 문자열 매핑. 테이블 출력 시 사용

### 시각화 전용 메타정보
- `RENDER_METAINFO`: 논문 그림 시각화 렌더링에 최적화된 클래스별 RGB 색상 설정 (예: `stop_line`(ID 9)은 명확한 구분을 위해 금색 `(255, 215, 0)`으로 지정)
- `RENDER_ID2BGR`: `RENDER_METAINFO`를 기반으로 자동 생성되는 BGR 색상 매핑 딕셔너리

### 기타
- `MODEL_PREFIX`: 모델 디렉토리 이름의 공통 접두사 (`"satellite_ade20k_250925_"`). `run_experiments.py`에서 CSV에 모델명 저장 시 이 접두사를 제거하는 데 사용

새로운 설정 작성 시 `src/backup/config-template.py`를 참고한다.

---

## 아키텍처

### 데이터 흐름

```
ade20k/images/validation/*.png                   ← 입력 위성 이미지
ade20k/annotations/validation/*.png             ← GT 팔레트 인덱스 레이블 이미지
Internimage/ (또는 mask2former/)
  └─ <model_name>/prediction/*.png              ← 세그멘테이션 모델 출력 (클래스별 색상 코딩)
        ↓
LineStringDetector.detect_lines()
        ↓
coco_pred_instances_origin.json                  # 초기 벡터화 결과 (skeletonization 직후)
coco_pred_instances_merge{1,2,3}.json            # 단계별 병합 결과 (merge3가 최종)
        ↓
[Table 생성]
num_params.py → num_params.csv                   # 모델별 파라미터 수
table_1.py    → table_1.csv                      # 알고리즘 변형별 성능 (AP, mIoU)
table_2.py    → table_2.csv                      # 차선 클래스별 성능 (AP20, mIoU)
              → table_3.csv                      # 차선 클래스별 심화 지표 (인스턴스 매칭 분석)
table_4.py    → table_4.csv                      # Ablation study
        ↓
[Figure 생성]
figure_1.py, figure_1_fin.py → Figure_1/, figure1.jpg  # 예측 마스크 개별/최종 비교 콜라주
figure_2.py                  → Figure_2/*.jpg          # 2x2 비교 콜라주 (원본/예측/초기선/최종선)
figure_3.py                  → Figure_3/*.png          # Guiding line & Safety zone 비교 콜라주
figure_4.py                  → Figure_4/*.png          # center_line 병합 중간과정 1x4 콜라주
figure_5.py                  → Figure_5/*.png          # 1x3 원본/GT/Prediction 비교 콜라주
figure_compare.py            → Figure_compare/*.png    # 2x2 GT/모델 3종 segmentation overlay 콜라주
```

## 런타임 디렉토리 구조

```
DATA_ROOT/
  ade20k/
    images/
      ├─ training/*.png
      ├─ validation/*.png                 # 원본 위성 이미지
      └─ test/*.png
    annotations/
      ├─ training/*.png
      ├─ validation/*.png                 # GT 팔레트 인덱스 레이블 이미지
      └─ test/*.png
    color_annotations/
      ├─ training/*.png
      ├─ validation/*.png                 # GT 컬러 시각화 레이블 이미지
      └─ test/*.png
  Internimage/
    ├─ checkpoint/                        # InternImage 체크포인트 (.pth)
    └─ satellite_ade20k_250925_internimage_large/
         └─ prediction/*.png              # 색상 코딩된 세그멘테이션 예측
  mask2former/
    ├─ checkpoint/                        # Mask2Former 체크포인트 (.pth)
    ├─ pre_trained/
    ├─ satellite_ade20k_250925_mask2former_large/
    │    └─ prediction/*.png
    └─ satellite_ade20k_250925_mask2former_small/
         └─ prediction/*.png
  results/
    ├─ merged_annotations.json             # COCO 형식 GT 어노테이션
    ├─ selected_annotation.json           # 평가용 GT 캐시: merged_annotations.json에서 EXCLUDE_IDS 클래스를 제거하고 COCO 평가 필드(id/iscrowd/area)를 보강한 것 (이미지 선별이 아니라 클래스 필터링). 원본보다 오래되면 자동 무효화 후 재생성됨
    ├─ num_params.csv                     # 모델별 파라미터 수
    ├─ total_performance.csv              # 하이퍼파라미터 전체 성능 탐색 결과
    ├─ <model_name>/
    │    └─ thick=T,stride=S,extend=E/     # 하이퍼파라미터 조합별 출력
    │         ├─ coco_pred_instances_origin.json
    │         ├─ coco_pred_instances_merge1.json
    │         ├─ coco_pred_instances_merge2.json
    │         ├─ coco_pred_instances_merge3.json
    │         └─ eval_result.csv
    ├─ Tables/
    │    ├─ table_1.csv
    │    ├─ table_2.csv
    │    ├─ table_3.csv
    │    └─ table_4.csv
    └─ Figure/
         ├─ Figure_1/                     # figure_1.py 출력 (개별 시각화 마스크)
         ├─ Figure_2/                     # figure_2.py 출력 (2x2 콜라주 [이미지명].jpg)
         ├─ Figure_3/                     # figure_3.py 출력 (Guiding/Safety [이미지명].png)
         ├─ Figure_4/                     # figure_4.py 출력 (center_line 병합과정 1x4 콜라주)
         ├─ Figure_5/                     # figure_5.py 출력 (1x3 원본-GT-Prediction 콜라주)
         ├─ Figure_compare/               # figure_compare.py 출력 (2x2 GT-모델3종 segmentation overlay 콜라주)
         └─ figure1.jpg                   # figure_1_fin.py 출력 (최종 세로형 콜라주)
```

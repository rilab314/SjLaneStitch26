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
5. 논문 작성을 위한 CSV 테이블 생성

---

## 실행 방법

모든 스크립트는 `src/` 디렉토리에서 실행해야 한다 (모듈 임포트가 상대 경로 기준):

```bash
cd src

# 단일 모델 설정으로 차선 감지 실행
python lane_detector.py

# 모든 모델과 파라미터 조합으로 전체 하이퍼파라미터 탐색 실행
python run_experiments.py

# 단일 실행 결과 평가 및 table_1.csv 생성
python Table/table_1.py

# 최적 모델/파라미터를 찾고 table_2.csv (클래스별 지표) 생성
python Table/table_2.py

# 논문용 Ablation Study 테이블 (table_3.csv) 생성
python Table/table_3.py

# 시각화 결과 생성
python Figure/figure_1_raw.py
python Figure/figure_2.py
```

---

## 설정

`src/config.py`에 모든 경로가 정의되어 있다. **실행 전 반드시 수정해야 한다:**
- `DATA_ROOT`: 데이터셋과 모델 출력이 저장된 루트 디렉토리
- `DATASET_PATH`: ADE20K 형식의 데이터셋 경로 (`DATA_ROOT/ade20k`)
- `MODEL_PATH`: 세그멘테이션 모델 예측 결과 디렉토리 (`DATA_ROOT/Internimage/...`)
- `RESULT_PATH`: JSON 예측 결과와 CSV 테이블이 저장되는 경로
- `COCO_MERGED_ANNO_PATH`: GT COCO 어노테이션 JSON 파일 경로 (`RESULT_PATH/merged_annotations.json`)

### 클래스 메타정보
- `METAINFO`: 클래스 ID, 이름, RGB 색상을 정의하는 딕셔너리 리스트. ID 0은 ignore, ID 1~11이 실제 차선 클래스
- `EXCLUDE_IDS`: 평가에서 제외할 클래스 ID 목록 (`[8, 10, 11]` — guiding_line, safety_zone, bicycle_lane)
- `EVAL_CLASS_IDS`: 실제 평가에 사용되는 클래스 ID 목록 (METAINFO에서 ID 0과 EXCLUDE_IDS를 제외하고 자동 생성)
- `ID2BGR`: 클래스 ID → BGR 색상 튜플 매핑. 예측 이미지에서 클래스별 픽셀을 추출할 때 사용
- `ID2NAME`: 클래스 ID → 클래스 이름 문자열 매핑. 테이블 출력 시 사용

### 기타
- `MODEL_PREFIX`: 모델 디렉토리 이름의 공통 접두사 (`"satellite_ade20k_250925_"`). `run_experiments.py`에서 CSV에 모델명 저장 시 이 접두사를 제거하는 데 사용

새로운 설정 작성 시 `src/config-template.py`를 참고한다.

---

## 아키텍처

### 데이터 흐름

```
ade20k/images/validation/*.png         ← 입력 위성 이미지
MODEL_PATH/prediction/*.png            ← 세그멘테이션 모델 출력 (클래스별 색상 코딩)
ade20k/annotations/validation/*.png   ← GT 레이블 이미지 (팔레트 인덱스 형식)
        ↓
LineStringDetector.detect_lines()
        ↓
coco_pred_instances_origin.json        # (c) 초기 벡터화 (skeletonization 직후)
coco_pred_instances_merge{1,2,3}.json  # 단계별 병합 결과 (merge3가 최종)
        ↓
table_1.py → table_1.csv               # 알고리즘 변형별 성능 (AP, mIoU)
table_2.py → table_2.csv               # 차선 클래스별 성능 (AP20, mIoU)
table_3.py → table_3.csv               # Ablation study
figure_2.py → Figure_2/*.jpg           # 2x2 시각화 콜라주
```

## 런타임 디렉토리 구조

```
DATA_ROOT/
  ade20k/
    images/validation/*.png               # 원본 위성 이미지
    annotations/validation/*.png          # GT 팔레트 인덱스 레이블 이미지
  Internimage/ (또는 Mask2Former/)
    <model_name>/
      prediction/*.png                    # 색상 코딩된 세그멘테이션 예측
  results/
    merged_annotations.json               # COCO 형식 GT 어노테이션
    num_params.csv                        # 모델별 파라미터 수
    <model_name>/
      thick=T,stride=S,extend=E/          # 하이퍼파라미터 조합별 출력
        coco_pred_instances_origin.json   # 초기 추출된 LineString
        coco_pred_instances_merge3.json   # 최종 병합된 LineString
        eval_result.csv                   # 해당 조합의 상세 성능
    Tables/
      table_1.csv, table_2.csv, table_3.csv
    Figure/
      Figure_1_raw/                       # Figure 1용 개별 시각화 결과
      Figure_2/                           # Figure 2용 2x2 콜라주 ([이미지명].jpg)
      figure1.jpg                         # Figure 1 최종 콜라주
```

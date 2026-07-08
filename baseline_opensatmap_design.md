# OpenSatMap Baseline 재구현 설계 (M1 비교용)

리뷰 지적 **M1**(외부 baseline 정량 비교 부재)에 답하기 위해, OpenSatMap 논문(NeurIPS 2024
D&B) §4.1의 baseline 후처리를 SjLaneStitch26 안에 재구현해 **동일 세그멘테이션 출력 위에서**
우리 파이프라인과 같은 metric으로 비교하는 설계다.

- 대상: **OpenSatMap watershed baseline만.** AerialLaneNet은 학습형 graph-detection(자기 backbone
  feature에 의존)이라 후처리 스왑이 불가 → 비교 제외, Related Work에서 개념 구분만.
- OpenSatMap 공식 저장소에는 baseline 코드가 없음("code soon", 데이터 준비 스크립트만) → **논문
  서술 기반 재구현**. 논문에서 인용한 OpenCV watershed 튜토리얼 레시피를 따른다.

---

## 1. 공정성 원칙 (반드시 고정)

같은 것 / 다른 것을 명확히 통제해, 차이가 **오직 후처리 알고리즘**에서 오도록 한다.

| 항목        | 값(고정)                                                      |
| --------- | ---------------------------------------------------------- |
| 세그멘테이션 입력 | Mask2Former (Swin-L) 예측 PNG (우리 최고 모델과 동일)                 |
| 평가 클래스    | 동일 9클래스(`EVAL_CLASS_IDS`, guiding/safety 제외)               |
| GT        | 동일 통합 어노테이션(`merged_annotations.json`)                     |
| 평가 하네스    | `evaluator.py`의 COCO AP@0.2 + mIoU 그대로                     |
| 래스터화 폭    | **3px**(우리 `convert_to_json`과 동일) — raster 폭이 아니라 알고리즘을 비교 |
| score     | 1.0 고정(우리와 동일, 단일 작동점)                                     |
| 클래스 처리    | 클래스별 독립(우리와 동일)                                            |

→ 두 방법 모두 "같은 seg 마스크 in → 폴리라인 인스턴스 out"이므로 층위가 정확히 일치.

**baseline은 논문 상수만 사용하고 validation에서 별도 튜닝하지 않는다**(그들 baseline의 "simple,
without bells and whistles" 취지 보존). 재현임을 본문에 명시: *"we re-implemented the OpenSatMap
baseline following their paper (§4.1); the official code was not released at the time."*

---

## 2. 재구현할 알고리즘 (OpenSatMap §4.1 baseline)

논문은 3단계로 분해: (1) semantic segmentation → (2) instance detection = **watershed** →
(3) instance vectorization = **denoise(≥100px) + point sampling**. (1)은 이미 있으므로 (2)(3)만
재구현한다.

의사코드(클래스별, 한 이미지):

```
입력: pred_img(색상 코딩 seg PNG), class_id
1) 이진 마스크  y_c = (pred_img == palette[class_id])            # 우리와 동일한 색 추출
2) 인스턴스 분리 (watershed, OpenCV 튜토리얼 레시피 = 논문 인용 [2]):
   - opening으로 노이즈 제거 (morph open, 3x3)
   - sure_bg = dilate(y_c)
   - dist   = distanceTransform(y_c, L2)
   - sure_fg = (dist > alpha * dist.max())                       # alpha: 재현 판단 지점 → §6
   - unknown = sure_bg - sure_fg
   - markers = connectedComponents(sure_fg); markers[unknown]=0
   - labels  = watershed(y_c_3ch, markers)                       # 각 라벨 = 한 인스턴스
3) 인스턴스별 벡터화 (h):
   for 각 인스턴스 라벨 L:
     mask_L = (labels == L)
     if area(mask_L) < 100: continue                             # 논문 상수: <100px 제거
     pts = 좌표(mask_L)                                          # (P,2)
     # sample-then-reconstruct: 선을 순서화해 폴리라인 구성
     주축 = PCA(pts).첫 성분
     order = argsort(pts @ 주축)                                 # 주축 투영으로 점 정렬
     poly = pts[order]
     poly = resample_polyline(poly, step=sample_stride)          # N점 균일 재샘플(우리 유틸 재사용)
     yield Strand(class_id=class_id, points=poly)
```

핵심: **우리 파이프라인의 강점 3종을 의도적으로 넣지 않는다.**

- ✗ 곡률 인지 추적 (PCA 정렬만)
- ✗ 단편 재연결(stitch/merge)
- ✗ 이중선 겹침 trim·평행 거부·residual 재추출
  → 그래서 단편화·이중선 붕괴가 그대로 남고, 우리 방법의 기여가 정량으로 드러남.

---

## 3. 우리 파이프라인과의 대조 (표로 정리해 본문/Related Work에 활용)

| 단계      | OpenSatMap baseline | 우리(LaneStitch)           |
| ------- | ------------------- | ------------------------ |
| 인스턴스 분리 | watershed           | Zhang–Suen 세선화 + 블롭 추적   |
| 순서화     | PCA 주축 정렬           | 곡률 패널티 추적(turn_penalty)  |
| 이중선     | 처리 없음(한 인스턴스로 남음)   | refinement로 겹침 trim, 대표선 |
| 누락 복원   | 없음                  | residual 재추출             |
| 단편      | 그대로                 | 끝점 확장 + 직렬 체이닝(merge×2)  |
| 노이즈     | <100px 제거           | min_lane_len + 평활화       |

---

## 4. SjLaneStitch26 이식 지점

**신규 파일 1개 + 실행/평가 스크립트 1개.** 기존 I/O·평가·래스터화는 전부 재사용.

### 4.1 `src/baseline_opensatmap.py` (신규)

- 클래스 `OpenSatMapBaseline`.
- 재사용 대상:
  - 세그 PNG 로드: `LaneStitcher._read_image`와 동일 경로 규칙(`/images/validation/`→
    `/prediction/`). 컴포지션으로 `LaneStitcher` 인스턴스를 하나 들고 `_read_image`·`_palette`·
    `convert_to_json`을 그대로 호출(래스터화 3px·RLE 인코딩 재사용 → 평가 정합 보장).
  - `resample_polyline`(모듈 함수) 그대로.
- 신규 로직: §2의 watershed + PCA 정렬(위 의사코드). `Strand`(id, class_id, points)만 채우면
  `convert_to_json`이 나머지 처리.
- 출력: `coco_pred_instances_baseline.json` 한 개(우리 `origin`/`merge*`와 나란히 저장).

### 4.2 `src/run_baseline.py` (신규)

- best 모델 경로 확정(`util.find_best_pred_json_path` 재사용) → `OpenSatMapBaseline.run()`으로
  validation 전량 처리 → baseline json 저장 → `evaluator.evaluate_coco_ap` +
  `evaluate_miou_json` 호출해 AP20/mIoU 출력.
- 우리 Merge×2 결과(기존 `coco_pred_instances_merge2.json`)와 한 줄로 대비 출력.

### 4.3 코드 규칙

- 저장소 `coding` 스킬 준수(OOP·길이 제한·네이밍). `OpenSatMapBaseline`은 단일 책임(후처리)만,
  파일 200줄 내 목표.

---

## 5. 평가·표 통합

- **평가:** baseline json을 기존 `evaluate_coco_ap`(catIds=EVAL_CLASS_IDS, IoU 0.2) +
  `evaluate_miou_json`에 그대로 투입 → 우리와 동일한 AP20·mIoU 산출.

- **표:** 새 소표 or Table 1 확장. 권장 형태(같은 seg 위 후처리 비교):
  
  | Method                          | Instances | AP20      | mIoU  |
  | ------------------------------- | --------- | --------- | ----- |
  | Segmentation (Swin-L)           | –         | –         | 39.09 |
  | OpenSatMap baseline (watershed) | ?         | ?         | ?     |
  | **Ours (Merge×2)**              | 36,896    | **43.29** | 36.66 |
  
  생성은 소형 `table_baseline.py`(baseline eval 결과 + 기존 best 행 조합). Table 4 상단에
  baseline 행을 끼우는 방식도 가능.

- **예상 결과:** watershed는 얇은 선에 약해 baseline AP20이 우리보다 낮게 나올 가능성이 큼(그게 그들
  baseline의 의도). 유리하게 나오더라도 "저자 최적 튜닝이 아닌 논문 서술 재현"임을 명시.

---

## 6. 재현 판단이 필요한 지점 (사용자 승인 요청)

논문 서술이 간결해 아래 두 가지는 내 기본 선택을 제안하되 확정이 필요하다.

1. **watershed 마커/전경 임계 `alpha`** — 얇은 선은 distanceTransform이 거의 평탄해 sure_fg가
   과도하게 깎임.
   
   - (기본 제안) 논문이 인용한 **OpenCV 튜토리얼 레시피 그대로**, `alpha=0.5`. 얇은 선에선 결국
     대부분 connected-component와 유사하게 동작(이중선 등 붙은 블롭은 한 인스턴스로 남음 = baseline
     한계 그대로 노출). → 가장 충실.
   - (대안) 마커를 못 세우는 클래스는 **connected components로 폴백**. 더 안정적이나 watershed
     취지에서 약간 벗어남.
   - **내 권장: 기본(충실 재현). 폴백은 쓰지 않음.**

2. **point sampling 순서화 방식** — "sample N points and connect"의 순서 규정이 없음.
   
   - (기본 제안) **PCA 주축 투영 정렬** — 선형 인스턴스에 단순·강건. 곡선/굽은 인스턴스는 다소
     왜곡되나 baseline이 곡률 처리를 안 한다는 점과 일치.
   - (대안) 인스턴스 스켈레톤화 후 끝점-끝점 순회. 곡선에 더 정확하지만 우리 세선화 이점을 baseline에
     섞게 됨(불공정 소지).
   - **내 권장: PCA 정렬**(baseline을 일부러 순진하게 유지).

3. **재샘플 간격** — baseline `sample_stride`를 우리 best(5px)와 **동일**하게 두어 기하 해상도 차이를
   제거(권장), 또는 논문식 고정 N. **내 권장: 5px 동일.**

---

## 7. 예상 공수·리스크

- 공수: 신규 코드 ~200줄 + 실행·표 스크립트. 반나절 규모.
- 리스크:
  - watershed 재현의 세부(위 §6-1)가 결과 수치를 좌우 → 승인 후 고정하고 본문에 레시피 명기.
  - baseline이 지나치게 약하게 나오면 "straw man" 인상 → 논문 상수 충실 재현·튜닝 배제로 방어.
  - 두 방법의 mIoU는 비슷하게(둘 다 seg 픽셀 기반) 나오고 **AP20에서 격차**가 날 것으로 예상 →
    "우리 기여는 커버리지가 아니라 객체 연속성 복원"이라는 본문 논지와 정합.

---

## 8. 다음 단계

§6의 세 판단(1: 충실 watershed / 2: PCA 정렬 / 3: 5px)에 대한 승인 → `baseline_opensatmap.py`·
`run_baseline.py` 작성 → validation 실행 → baseline AP20/mIoU 확보 → Table·본문 반영.

import os
import glob
import csv
import pandas as pd
import config as cfg


def parse_value(val):
    """문자열 데이터를 숫자(float/int)로 변환, 실패 시 0 반환"""
    try:
        # '-' 등의 문자가 들어올 경우 0으로 처리
        if val == '-' or val == '':
            return 0.0
        return float(val)
    except ValueError:
        return 0.0


def summarize_experiments():
    base_path = cfg.RESULT_PATH

    # 1. 모든 table_1.csv 파일 찾기 (재귀적 탐색)
    # 패턴: RESULT_PATH/thickness=*/sample_stride=*/extend_len=*/table_1.csv
    search_pattern = os.path.join(base_path, "thickness=*", "sample_stride=*", "extend_len=*", "table_1.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"No results found in {base_path}. Please run 'run_experiments.py' first.")
        return

    summary_data = []

    print(f"Found {len(csv_files)} experiment results. Processing...")

    for file_path in csv_files:
        try:
            # 2. 경로에서 파라미터 추출
            # os.sep을 사용하여 OS 환경에 관계없이 경로 분리
            parts = file_path.split(os.sep)

            # 파라미터 파싱 (폴더명에서 숫자만 추출)
            thickness = int(parts[-4].split('=')[-1])
            stride = int(parts[-3].split('=')[-1])
            extend = int(parts[-2].split('=')[-1])

            # 3. CSV 파일 읽기 및 데이터 추출
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

                target_row = None
                for row in rows:
                    if row and "linestrings merged" in row[0]:
                        target_row = row
                        break

                if target_row:
                    instances = int(parse_value(target_row[1]))
                    ap20 = parse_value(target_row[3])

                    summary_data.append({
                        'Thickness': thickness,
                        'Stride': stride,
                        'Extend': extend,
                        'AP20 (all)': ap20,
                        'Instances': instances,
                        'Path': os.path.dirname(file_path)
                    })
        except Exception as e:
            print(f"[Warning] Failed to parse {file_path}: {e}")

    # 4. 데이터 프레임 생성 및 정렬
    if not summary_data:
        print("No valid data extracted.")
        return

    df = pd.DataFrame(summary_data)

    # ---------------------------------------------------------
    # [수정 부분] 파라미터가 작은 순서대로 정렬 (오름차순)
    # ---------------------------------------------------------
    df_sorted = df.sort_values(by=['Thickness', 'Stride', 'Extend'], ascending=True)

    # 보기 좋게 컬럼 순서 재배치
    cols = ['Thickness', 'Stride', 'Extend', 'AP20 (all)', 'Instances']
    df_display = df_sorted[cols]

    # 5. 결과 출력
    print("\n" + "=" * 60)
    print(" Experiment Summary (Sorted by Parameters)")
    print("=" * 60)
    print(df_display.to_string(index=False))
    print("=" * 60)

    # 6. CSV 파일로 저장
    save_path = os.path.join(base_path, "summary_report.csv")
    df_sorted.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n[Done] Summary saved to: {save_path}")

    # 7. 성능 기준 최적의 파라미터 별도 출력 (참고용)
    best_df = df.sort_values(by=['AP20 (all)', 'Instances'], ascending=[False, False])
    best = best_df.iloc[0]
    print(f"\n[Best Performance Reference]")
    print(f" >> Thickness={int(best['Thickness'])}, Stride={int(best['Stride'])}, Extend={int(best['Extend'])}")
    print(f" >> AP20: {best['AP20 (all)']:.6f}, Instances: {int(best['Instances'])}")


if __name__ == "__main__":
    try:
        summarize_experiments()
    except ImportError:
        print("Pandas library is required for this script.")
        print("Please install it using: pip install pandas")
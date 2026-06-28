import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


class Table4Builder:
    ABLATION_PARAMS = ['sample_strides', 'extend_lens', 'turn_penalties']
    METRICS = ['instances', 'AP10', 'AP20', 'AP50', 'mIoU']

    def __init__(self, total_csv_path, save_path):
        self.total_csv_path = total_csv_path
        self.save_path = save_path
        self.df = pd.read_csv(total_csv_path)

    def build(self):
        model_name, merge_count = self._find_best()
        result = self._build_ablation_table(model_name, merge_count)
        self._save(result, model_name, merge_count)

    def _find_best(self):
        detector_df = self.df[self.df['instances'] > 0]
        best_row = detector_df.loc[detector_df['AP20'].idxmax()]
        model_name = best_row['model_name']
        merge_count = int(best_row['merge_count'])
        print(f"Ablation 고정값 — model={model_name}, merge_count={merge_count} "
              f"(AP20={best_row['AP20']:.4f})")
        return model_name, merge_count

    def _build_ablation_table(self, model_name, merge_count):
        mask = (
            (self.df['model_name'] == model_name) &
            (self.df['merge_count'] == merge_count) &
            (self.df['instances'] > 0)
        )
        model_df = self.df[mask].copy()
        cols = self.ABLATION_PARAMS + self.METRICS
        return model_df[cols].sort_values(self.ABLATION_PARAMS).reset_index(drop=True)

    def _save(self, result, model_name, merge_count):
        int_cols = ['sample_strides', 'extend_lens', 'turn_penalties', 'instances']
        result[int_cols] = result[int_cols].astype(int)
        metric_cols = ['AP10', 'AP20', 'AP50', 'mIoU']
        result[metric_cols] = (result[metric_cols] * 100).round(2)  # % 단위로 변환
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        result.to_csv(self.save_path, index=False, encoding='utf-8')
        print(f"\nAblation Study — model: {model_name}, merge_count: {merge_count} (고정)")
        print(f"Saved to: {self.save_path}")
        print(result.to_string(index=False))
        print('table 4 shape:', result.shape)


def main():
    total_csv_path = os.path.join(cfg.RESULT_PATH, 'total_performance.csv')
    save_path = os.path.join(cfg.RESULT_PATH, 'Tables', 'table_4.csv')
    Table4Builder(total_csv_path, save_path).build()


if __name__ == '__main__':
    main()

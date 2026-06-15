import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


class Table3Builder:
    ABLATION_PARAMS = ['merge_count', 'sample_strides', 'extend_lens']
    METRICS = ['instances', 'AP20', 'mIoU']

    def __init__(self, total_csv_path, save_path):
        self.total_csv_path = total_csv_path
        self.save_path = save_path
        self.df = pd.read_csv(total_csv_path)

    def build(self):
        model_name = self._find_best_model()
        result = self._build_ablation_table(model_name)
        self._save(result, model_name)

    def _find_best_model(self):
        detector_df = self.df[self.df['instances'] > 0]
        best_row = detector_df.loc[detector_df['AP20'].idxmax()]
        best_model = best_row['model_name']
        print(f"Best model for ablation: {best_model} (AP20={best_row['AP20']:.4f})")
        return best_model

    def _build_ablation_table(self, model_name):
        model_df = self.df[
            (self.df['model_name'] == model_name) &
            (self.df['instances'] > 0)
        ].copy()
        cols = self.ABLATION_PARAMS + self.METRICS
        return model_df[cols].sort_values(self.ABLATION_PARAMS).reset_index(drop=True)

    def _save(self, result, model_name):
        result[['merge_count', 'instances']] = result[['merge_count', 'instances']].astype(int)
        result[['AP20', 'mIoU']] = (result[['AP20', 'mIoU']] * 100).round(2)  # % 단위로 변환
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        result.to_csv(self.save_path, index=False, encoding='utf-8')
        print(f"\nAblation Study — model: {model_name}")
        print(f"Saved to: {self.save_path}")
        print(result.to_string(index=False))
        print('table 3 shape:', result.shape)


def main():
    total_csv_path = os.path.join(cfg.RESULT_PATH, 'total_performance.csv')
    save_path = os.path.join(cfg.RESULT_PATH, 'Tables', 'table_3.csv')
    Table3Builder(total_csv_path, save_path).build()


if __name__ == '__main__':
    main()

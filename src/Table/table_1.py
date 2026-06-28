import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


class Table1Builder:
    def __init__(self, total_csv_path, save_path, num_params_path):
        self.total_csv_path = total_csv_path
        self.save_path = save_path
        self.df = pd.read_csv(total_csv_path)
        self.num_params_df = pd.read_csv(num_params_path)

    def build(self):
        thickness, stride, extend_len, turn = self._find_best_params()
        result = self._filter_and_sort(thickness, stride, extend_len, turn)
        result.drop_duplicates(inplace=True, ignore_index=True)
        print('intermediate result\n', result.to_string(index=False))
        result = self._add_params_column(result)
        self._save(result)

    def _find_best_params(self):
        best_row = self.df.loc[self.df['AP20'].idxmax()]
        thickness = best_row['thicknesses']
        stride = best_row['sample_strides']
        extend_len = best_row['extend_lens']
        turn = best_row['turn_penalties']
        print(f"Best params — thicknesses={thickness}, sample_strides={stride}, "
              f"extend_lens={extend_len}, turn_penalties={turn}, max AP20={best_row['AP20']:.6f}")
        return thickness, stride, extend_len, turn

    def _filter_and_sort(self, thickness, stride, extend_len, turn):
        algo_mask = (
            (self.df['thicknesses'] == thickness) &
            (self.df['sample_strides'] == stride) &
            (self.df['extend_lens'] == extend_len) &
            (self.df['turn_penalties'] == turn)
        )
        filtered = self.df[algo_mask].copy()
        return filtered.sort_values(
            by=['model_name', 'merge_count'],
            ascending=[True, True],
            na_position='first'
        ).reset_index(drop=True)

    def _add_params_column(self, df):
        params_map = self.num_params_df.set_index('model')['total_params_M']
        df['params(M)'] = df['model_name'].map(params_map)
        return df

    def _save(self, result):
        int_cols = ['merge_count', 'thicknesses', 'sample_strides', 'extend_lens', 'instances']
        result[int_cols] = result[int_cols].astype("Int64")
        metric_cols = ['AP10', 'AP20', 'AP50', 'mIoU']
        result[metric_cols] = (result[metric_cols] * 100).round(2)  # % 단위로 변환
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        result.to_csv(self.save_path, index=False, encoding='utf-8')
        print(f"Table 1 saved to: {self.save_path}")
        print(result.to_string(index=False))


def main():
    total_csv_path = os.path.join(cfg.RESULT_PATH, 'total_performance.csv')
    save_path = os.path.join(cfg.RESULT_PATH, 'Tables', 'table_1.csv')
    num_params_path = os.path.join(cfg.RESULT_PATH, 'num_params.csv')
    Table1Builder(total_csv_path, save_path, num_params_path).build()


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class DataPreprocessingWorkflow:
    """教学友好的数据预处理工作流。

    这个类将“读取数据、诊断缺失、执行填补、标准化、可视化和导出结果”
    串联为一套完整流程，同时保留清晰的中文注释，方便后续同学直接复用。
    """

    def __init__(
        self,
        input_path: str = "tax_data.csv",
        output_path: str = "clean_tax_data.csv",
        figure_path: str = "standardization_comparison.png",
        missing_threshold: float = 0.10,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.figure_path = Path(figure_path)
        self.missing_threshold = missing_threshold

        self.raw_df: Optional[pd.DataFrame] = None
        self.imputed_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None

        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.imputation_summary: List[Dict[str, object]] = []

    def load_data(self) -> pd.DataFrame:
        """读取原始数据，并识别字段类型。"""
        try:
            # 第一步先确保数据被正确读入。
            # 如果源文件读取失败，后面的缺失值处理和标准化都无法开展。
            df = pd.read_csv(self.input_path)

            if df.empty:
                raise ValueError("输入数据为空，无法继续执行。")

            self.raw_df = df
            self.numeric_columns = df.select_dtypes(include="number").columns.tolist()
            self.categorical_columns = [col for col in df.columns if col not in self.numeric_columns]

            print(f"[Step 1] 成功读取数据：{self.input_path}")
            print(f"[信息] 数据形状：{df.shape}")
            print(f"[信息] 数值型字段：{self.numeric_columns}")
            print(f"[信息] 类别型字段：{self.categorical_columns}")
            return df
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"未找到输入文件：{self.input_path}") from exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"文件存在，但内容为空：{self.input_path}") from exc
        except Exception as exc:
            raise RuntimeError(f"读取数据时发生异常：{exc}") from exc

    def diagnose_missing_values(self) -> pd.DataFrame:
        """输出缺失值诊断报告。"""
        if self.raw_df is None:
            raise ValueError("请先执行 load_data()。")

        try:
            # 先做缺失诊断，而不是直接填补。
            # 这样学生可以先看到“问题有多严重”，再理解为什么要选某种策略。
            report = pd.DataFrame(
                {
                    "字段名": self.raw_df.columns,
                    "数据类型": self.raw_df.dtypes.astype(str).values,
                    "缺失数量": self.raw_df.isna().sum().values,
                    "缺失比例": self.raw_df.isna().mean().round(4).values,
                }
            )

            print("\n[Step 2] 缺失值诊断报告")
            print(report.to_string(index=False))
            return report
        except Exception as exc:
            raise RuntimeError(f"生成缺失值诊断报告时发生异常：{exc}") from exc

    def build_imputation_summary(self) -> pd.DataFrame:
        """根据缺失比例生成填补策略说明。"""
        if self.raw_df is None:
            raise ValueError("请先执行 load_data()。")

        try:
            self.imputation_summary = []

            for column in self.numeric_columns:
                missing_ratio = float(self.raw_df[column].isna().mean())
                non_missing = self.raw_df[column].dropna()

                # 这里采用“按缺失比例选择填补策略”的规则：
                # 缺失较少时，用均值保留整体水平；
                # 缺失较多时，用中位数增强稳健性，减少异常值影响。
                if non_missing.empty:
                    strategy = "constant"
                    reason = "该列全部缺失，均值和中位数无法计算，因此退化为常数 0 填补。"
                elif missing_ratio >= self.missing_threshold:
                    strategy = "median"
                    reason = (
                        f"缺失比例为 {missing_ratio:.2%}，达到阈值 {self.missing_threshold:.0%}，"
                        "优先使用中位数，降低极端值的干扰。"
                    )
                else:
                    strategy = "mean"
                    reason = (
                        f"缺失比例为 {missing_ratio:.2%}，低于阈值 {self.missing_threshold:.0%}，"
                        "优先使用均值，保持整体平均水平。"
                    )

                self.imputation_summary.append(
                    {
                        "字段名": column,
                        "字段类型": "数值型",
                        "缺失比例": round(missing_ratio, 4),
                        "填补策略": strategy,
                        "策略说明": reason,
                    }
                )

            for column in self.categorical_columns:
                missing_ratio = float(self.raw_df[column].isna().mean())

                # 类别型特征不能直接做均值或中位数填补，
                # 使用众数可以尽量保持原有的业务语义。
                self.imputation_summary.append(
                    {
                        "字段名": column,
                        "字段类型": "类别型",
                        "缺失比例": round(missing_ratio, 4),
                        "填补策略": "most_frequent",
                        "策略说明": "类别型字段使用众数填补，以保留最常见的类别信息。"
                        if missing_ratio > 0
                        else "该列无缺失值，保持原样。",
                    }
                )

            summary_df = pd.DataFrame(self.imputation_summary)
            print("\n[Step 2] 缺失值处理策略")
            print(summary_df.to_string(index=False))
            return summary_df
        except Exception as exc:
            raise RuntimeError(f"生成填补策略说明时发生异常：{exc}") from exc

    def impute_missing_values(self) -> pd.DataFrame:
        """执行缺失值填补。"""
        if self.raw_df is None:
            raise ValueError("请先执行 load_data()。")
        if not self.imputation_summary:
            self.build_imputation_summary()

        try:
            df = self.raw_df.copy()

            numeric_groups: Dict[str, List[str]] = {
                "mean": [],
                "median": [],
                "constant": [],
            }

            for item in self.imputation_summary:
                if item["字段类型"] == "数值型":
                    numeric_groups[str(item["填补策略"])].append(str(item["字段名"]))

            if numeric_groups["mean"]:
                mean_imputer = SimpleImputer(strategy="mean")
                df[numeric_groups["mean"]] = mean_imputer.fit_transform(df[numeric_groups["mean"]])

            if numeric_groups["median"]:
                median_imputer = SimpleImputer(strategy="median")
                df[numeric_groups["median"]] = median_imputer.fit_transform(df[numeric_groups["median"]])

            if numeric_groups["constant"]:
                constant_imputer = SimpleImputer(strategy="constant", fill_value=0)
                df[numeric_groups["constant"]] = constant_imputer.fit_transform(
                    df[numeric_groups["constant"]]
                )

            if self.categorical_columns:
                categorical_imputer = SimpleImputer(strategy="most_frequent")
                df[self.categorical_columns] = categorical_imputer.fit_transform(df[self.categorical_columns])

            self.imputed_df = df

            print("\n[Step 2] 缺失值填补完成")
            print(self.imputed_df.isna().sum().to_string())
            return df
        except Exception as exc:
            raise RuntimeError(f"缺失值填补时发生异常：{exc}") from exc

    def scale_numeric_features(self) -> pd.DataFrame:
        """对数值型字段进行 Z-score 标准化。"""
        if self.imputed_df is None:
            raise ValueError("请先执行 impute_missing_values()。")

        try:
            # 这里保留原始字段，同时新增 *_zscore 字段。
            # 这样既能保留业务原值，也能得到模型友好的标准化特征，方便课堂对比。
            cleaned_df = self.imputed_df.copy()

            if not self.numeric_columns:
                self.cleaned_df = cleaned_df
                print("\n[Step 3] 当前数据中没有数值型字段，跳过标准化。")
                return cleaned_df

            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(self.imputed_df[self.numeric_columns])
            scaled_columns = [f"{column}_zscore" for column in self.numeric_columns]
            scaled_df = pd.DataFrame(scaled_values, columns=scaled_columns, index=self.imputed_df.index)

            self.cleaned_df = pd.concat([cleaned_df, scaled_df], axis=1)

            print("\n[Step 3] 标准化完成")
            print(f"[信息] 新增标准化字段：{scaled_columns}")
            print("[信息] 标准化后均值（应接近 0）：")
            print(self.cleaned_df[scaled_columns].mean().round(4).to_string())
            print("[信息] 标准化后标准差（应接近 1）：")
            print(self.cleaned_df[scaled_columns].std(ddof=0).round(4).to_string())
            return self.cleaned_df
        except Exception as exc:
            raise RuntimeError(f"特征标准化时发生异常：{exc}") from exc

    def visualize_distributions(self) -> Path:
        """绘制处理前后的分布对比图。"""
        if self.raw_df is None or self.cleaned_df is None:
            raise ValueError("请先完成数据读取与标准化。")

        try:
            if not self.numeric_columns:
                print("\n[Step 4] 没有数值型字段，跳过可视化。")
                return self.figure_path

            # 在中文教学环境里，提前设置字体可以减少图表乱码问题。
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            sns.set_theme(style="whitegrid")

            row_count = len(self.numeric_columns)
            fig, axes = plt.subplots(row_count, 4, figsize=(20, 4.5 * row_count))

            if row_count == 1:
                axes = [axes]

            for row_index, column in enumerate(self.numeric_columns):
                scaled_column = f"{column}_zscore"
                current_axes = axes[row_index]

                # 直方图用于观察分布形态和中心位置变化。
                sns.histplot(self.raw_df[column].dropna(), kde=True, color="#4C72B0", ax=current_axes[0])
                current_axes[0].set_title(f"{column} - 处理前直方图")

                sns.histplot(
                    self.cleaned_df[scaled_column].dropna(),
                    kde=True,
                    color="#55A868",
                    ax=current_axes[1],
                )
                current_axes[1].set_title(f"{scaled_column} - 标准化后直方图")

                # 箱线图用于观察中位数、四分位数和潜在异常值。
                sns.boxplot(x=self.raw_df[column].dropna(), color="#4C72B0", ax=current_axes[2])
                current_axes[2].set_title(f"{column} - 处理前箱线图")

                sns.boxplot(x=self.cleaned_df[scaled_column].dropna(), color="#55A868", ax=current_axes[3])
                current_axes[3].set_title(f"{scaled_column} - 标准化后箱线图")

            plt.tight_layout()
            plt.savefig(self.figure_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"\n[Step 4] 分布对比图已保存：{self.figure_path}")
            return self.figure_path
        except Exception as exc:
            raise RuntimeError(f"可视化时发生异常：{exc}") from exc

    def save_clean_data(self) -> Path:
        """保存清洗后的结果数据。"""
        if self.cleaned_df is None:
            raise ValueError("请先完成清洗流程，再保存结果。")

        try:
            # 使用 utf-8-sig 方便在 Windows 环境下被 Excel 正常打开中文列名。
            self.cleaned_df.to_csv(self.output_path, index=False, encoding="utf-8-sig")
            print(f"\n[Step 5] 清洗后的数据已保存：{self.output_path}")
            return self.output_path
        except Exception as exc:
            raise RuntimeError(f"保存清洗后数据时发生异常：{exc}") from exc

    def run(self) -> pd.DataFrame:
        """按顺序执行完整工作流。"""
        try:
            self.load_data()
            self.diagnose_missing_values()
            self.build_imputation_summary()
            self.impute_missing_values()
            self.scale_numeric_features()
            self.visualize_distributions()
            self.save_clean_data()

            print("\n[完成] 数据预处理与可视化工作流执行成功。")
            return self.cleaned_df if self.cleaned_df is not None else pd.DataFrame()
        except Exception as exc:
            print(f"\n[错误] 工作流执行失败：{exc}")
            raise


def build_default_workflow(
    input_path: str = "tax_data.csv",
    output_path: str = "clean_tax_data.csv",
    figure_path: str = "standardization_comparison.png",
    missing_threshold: float = 0.10,
) -> DataPreprocessingWorkflow:
    """创建一个默认工作流实例，方便学生在 Notebook 中直接调用。"""
    return DataPreprocessingWorkflow(
        input_path=input_path,
        output_path=output_path,
        figure_path=figure_path,
        missing_threshold=missing_threshold,
    )


def run_tax_workflow(
    input_path: str = "tax_data.csv",
    output_path: str = "clean_tax_data.csv",
    figure_path: str = "standardization_comparison.png",
    missing_threshold: float = 0.10,
) -> pd.DataFrame:
    """一键执行默认税收案例工作流。"""
    workflow = build_default_workflow(
        input_path=input_path,
        output_path=output_path,
        figure_path=figure_path,
        missing_threshold=missing_threshold,
    )
    return workflow.run()


if __name__ == "__main__":
    run_tax_workflow()

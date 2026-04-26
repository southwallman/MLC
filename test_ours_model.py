import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入你的核心组件
from paper_ablation_MTI_HANet import get_paper_ablation_model
from testv7_2 import (
    load_preloaded_test_data,
    TestDataset,
    test_with_best_params_from_json_ai,
    GLOBAL_LABELS,
    _config_namespace_from_json_best_params
)


def batch_test_ours_folder(root_dir, test_data_path, output_csv="ours_batch_results.csv"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 预加载测试集 (全局只读一次，提速核心)
    print(f"📦 正在加载测试集: {test_data_path}")
    preloaded = load_preloaded_test_data(test_data_path)
    test_loader = DataLoader(
        TestDataset(preloaded["images"], preloaded["targets"], preloaded["paths"]),
        batch_size=8, shuffle=False, num_workers=0, drop_last=True, pin_memory=True
    )

    # 2. 扫描文件夹下所有的 trial 权重
    weight_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".pth")]
    print(f"🔍 找到 {len(weight_files)} 个 Ours 模型权重文件。")

    all_metrics = []

    # 3. 开始循环测试
    for weight_path in tqdm(weight_files, desc="Ours Testing"):
        # 加载权重并提取保存的 config
        try:
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
            cfg_dict = ckpt.get("config", {})

            # 构建模型配置 (ours 对应 MTI_HANet_Ours 架构)
            config = _config_namespace_from_json_best_params(
                cfg_dict,
                num_classes=len(GLOBAL_LABELS),
                batch_size=8,
                device=device
            )

            # 实例化 Ours 模型 (MTI-HANet)
            model = get_paper_ablation_model("ours", GLOBAL_LABELS, config)

            # 执行测试 (内部含动态阈值搜索)
            # 注意：该函数会生成每个 trial 的详细 csv，我们在这里通过它的返回值收集整体指标
            df_res, _, _ = test_with_best_params_from_json_ai(
                model=model,
                name=os.path.basename(weight_path),
                model_save_path=weight_path,
                test_loader=test_loader,
                output_dir="ours_test_results",
                label_list=GLOBAL_LABELS
            )

            # 提取 "Overall" 行的数据用于汇总
            overall_stats = df_res[df_res['Label'] == 'Overall'].iloc[0].to_dict()
            overall_stats['file_name'] = os.path.basename(weight_path)
            all_metrics.append(overall_stats)

            # 及时释放内存
            del model, ckpt
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ 文件 {weight_path} 测试出错: {e}")
            continue

    # 4. 汇总导出最终对比表
    if all_metrics:
        final_df = pd.DataFrame(all_metrics)
        # 按照 F1 或 mAP 降序排列，帮你直接锁定表现最好的 Trial
        final_df = final_df.sort_values(by='F1', ascending=False)
        final_df.to_csv(output_csv, index=False, encoding='utf_8_sig')
        print(f"✅ 批量测试完成！汇总表已保存至: {output_csv}")


if __name__ == "__main__":
    # --- 配置区域 ---
    TARGET_FOLDER = "./optuna_trial_records_mode4_ours_ITDD_20260422_204555"  # 你图片中的目录
    TEST_DATA = "../dataset/shezhenv3_test_data.pth"

    batch_test_ours_folder(TARGET_FOLDER, TEST_DATA)
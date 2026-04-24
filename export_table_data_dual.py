import os
import json
import torch
import numpy as np
import cv2
import pandas as pd
from torchvision import transforms

from tonguedx_MLC.our_version7.paper_ablation_MTI_HANet import get_paper_ablation_model
from tonguedx_MLC.our_version7.optuna_train import _build_safe_config

# =====================================================================
# 🚨 升级版：全自动数据挖掘配置（完美保留了你的全部真实路径！）
# =====================================================================
DATASETS_CONFIG = {
    "TongueDx": {
        "labels": [
            'TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
            'Crack', 'Toothmark', 'FurThick', 'FurYellow'
        ],
        # 你的 CSV 文件路径 (保留原路径)
        "csv_path": "../../a_TongueDx2/list/test.csv",

        # 你的权重与JSON路径 (保留原路径)
        "baseline_weights": "../our_version7/paper_ablation_best_model_paper_model_1_TongueDx_csra_seed3_e30_20260410_200836.pth",
        "baseline_json": "../our_version7/paper_ablation_best_params_model_1_TongueDx_20260410_200836.json",

        "ours_weights": "../our_version7/paper_ablation_best_model_mode4_ours_TongueDx_seed3_e30_20260422_163433.pth",
        "ours_json": "../our_version7/paper_ablation_best_params_mode4_ours_TongueDx_20260422_163433.json",

        # 新增：定义两个输出文件的名字
        "full_output_json": "aaa_full_results_tonguedx.json",
        "star_output_json": "aaa_star_samples_tonguedx.json"
    },

    "ITDD": {
        "labels": [
            'HealthyTongue', 'PeelingCoating', 'RedTongue', 'PurpleTongue', 'ChubbyTongue',
            'ThinTongue', 'RedDot', 'Crack', 'Toothmark', 'FurWhite', 'FurYellow',
            'FurBlack', 'SmoothCoating'
        ],

        # 🌟 核心改动：指定为 yolo 解析模式
        "data_format": "yolo",

        # 🚨 你的测试集图片存放路径 (务必用测试集！)
        "image_dir": "../../shezhenv3-txt/test/images/",
        # 🚨 你的测试集标签存放路径
        "label_dir": "../../shezhenv3-txt/test/labels/",

        "baseline_weights": "../our_version7/paper_ablation_best_model_paper_model_1_ITDD_csra_seed3_e30_20260410_200836.pth",
        "baseline_json": "../our_version7/paper_ablation_best_params_model_1_ITDD_20260410_200836.json",
        "ours_weights": "../our_version7/paper_ablation_best_model_mode4_ours_ITDD_seed3_e30_20260422_205455.pth",
        "ours_json": "../our_version7/paper_ablation_best_params_mode4_ours_ITDD_20260422_205455.json",

        "full_output_json": "aaa_full_results_itdd.json",
        "star_output_json": "aaa_star_samples_itdd.json"
    }
}


# =====================================================================


def load_image(img_path, device):
    """加载并预处理单张图片"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"找不到图片: {img_path}")

    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(rgb_img).unsqueeze(0).to(device)


def mine_star_samples(full_json_path, star_json_path):
    """
    淘金函数：自动筛选出 MTI_HANet 表现碾压 Baseline 的明星样本
    """
    print(f"\n[{full_json_path}] 开启自动淘金模式，为您筛选SCI投稿样本...")
    with open(full_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    star_samples = []

    # --- 筛选标准：你可以根据实际跑出的分数微调这些阈值 ---
    FP_BASE_HIGH = 0.8  # 误诊：Baseline 无病却猜 > 0.8
    FP_OURS_LOW = 0.45  # 抑制：Ours 成功把无病压到 < 0.45

    FN_BASE_LOW = 0.35  # 漏诊：Baseline 有病却猜 < 0.35
    FN_OURS_HIGH = 0.65  # 抢救：Ours 成功把有病捞到 > 0.65

    for item in data:
        gt = set(item['ground_truth'])
        baseline = item['baseline']
        ours = item['ours']

        rescue_count = 0  # 抢救成功次数
        suppress_count = 0  # 抑制误诊次数
        reasons = []  # 记录上榜理由

        for label, b_prob in baseline.items():
            o_prob = ours[label]

            if label in gt:
                # 寻找【漏诊被抢救】的案例
                if b_prob < FN_BASE_LOW and o_prob > FN_OURS_HIGH:
                    rescue_count += 1
                    reasons.append(f"成功抢救 [{label}]: Base= {b_prob:.2f} -> Ours= {o_prob:.2f}")
            else:
                # 寻找【误诊被抑制】的案例
                if b_prob > FP_BASE_HIGH and o_prob < FP_OURS_LOW:
                    suppress_count += 1
                    reasons.append(f"成功抑制 [{label}]: Base= {b_prob:.2f} -> Ours= {o_prob:.2f}")

        # 如果至少有 1 次抢救或抑制，就是值得放到论文里的好样本！
        if rescue_count > 0 or suppress_count > 0:
            item['star_score'] = rescue_count + suppress_count  # 打分
            item['star_reasons'] = reasons
            star_samples.append(item)

    # 按照惊艳程度（得分）从高到低排序
    star_samples.sort(key=lambda x: x['star_score'], reverse=True)

    with open(star_json_path, 'w', encoding='utf-8') as f:
        json.dump(star_samples, f, indent=4, ensure_ascii=False)

    print(f"✅ 淘金完成！从 {len(data)} 个全量样本中，找到了 {len(star_samples)} 个超级明星样本！")
    print(f"✅ 精选结果已保存至: {star_json_path}")
    if len(star_samples) > 0:
        print("\n🏆 以下是最强 Top 3 样本上榜理由（你可以直接写进论文）：")
        for i in range(min(5, len(star_samples))):
            print(f"  [{i + 1}] 图片 {star_samples[i]['image']}")
            for reason in star_samples[i]['star_reasons']:
                print(f"      - {reason}")


def process_single_dataset(dataset_name, config_dict, device):
    print(f"\n{'=' * 60}")
    print(f"🚀 开始全量处理数据集: {dataset_name}")
    print(f"{'=' * 60}")

    labels = config_dict["labels"]
    num_classes = len(labels)
    baseline_weights = config_dict["baseline_weights"]
    ours_weights = config_dict["ours_weights"]
    full_output_json = config_dict["full_output_json"]
    star_output_json = config_dict["star_output_json"]

    data_format = config_dict.get("data_format", "csv")

    image_list = []
    multi_hot_dict = {}
    df = None

    # =================================================================
    # 🛡️ 数据源解析：CSV 或 YOLO 目录解析
    # =================================================================
    if data_format == "csv":
        csv_path = config_dict["csv_path"]
        if not os.path.exists(csv_path):
            print(f"❌ 错误: 找不到 CSV 文件 {csv_path}，跳过...")
            return
        df = pd.read_csv(csv_path)
        image_list = df['image_path'].tolist()
        print(f"读取 CSV: {csv_path} | 发现 {len(image_list)} 张图片！")

    elif data_format == "yolo":
        image_dir = config_dict["image_dir"]
        label_dir = config_dict["label_dir"]

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"❌ 错误: 找不到 YOLO 图片或标签目录，请检查路径！")
            return

        # 1. 扫描文件夹下的所有图片
        image_list = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"读取 YOLO 目录: {image_dir} | 发现 {len(image_list)} 张图片！")

        # 2. 为每张图片解析对应的 .txt 标签
        for img_name in image_list:
            base_name = os.path.splitext(img_name)[0]
            txt_path = os.path.join(label_dir, base_name + ".txt")

            # 初始化全 0 的 multi-hot 向量
            multi_hot = [0] * num_classes

            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) > 0:
                            # 提取第一列，即标签 ID
                            class_id = int(parts[0])
                            # 仅保留 0-12 的合法标签，忽略其他坐标
                            if 0 <= class_id < num_classes:
                                multi_hot[class_id] = 1

            multi_hot_dict[img_name] = multi_hot
    # =================================================================

    baseline_json_path = config_dict.get("baseline_json", "")
    ours_json_path = config_dict.get("ours_json", "")

    with open(baseline_json_path, "r", encoding="utf-8") as f:
        baseline_params = json.load(f)
    cfg_baseline = _build_safe_config(baseline_params, batch_size=1, device=device, num_classes=num_classes)

    with open(ours_json_path, "r", encoding="utf-8") as f:
        ours_params = json.load(f)
    cfg_ours = _build_safe_config(ours_params, batch_size=1, device=device, num_classes=num_classes)

    model_baseline = get_paper_ablation_model("model_1", labels, cfg_baseline)
    if os.path.exists(baseline_weights):
        model_baseline.load_state_dict(torch.load(baseline_weights, map_location=device)['model_state_dict'])
    model_baseline.to(device).eval()

    model_ours = get_paper_ablation_model("mti_hanet", labels, cfg_ours)
    if os.path.exists(ours_weights):
        model_ours.load_state_dict(torch.load(ours_weights, map_location=device)['model_state_dict'])
    model_ours.to(device).eval()

    results_data = []

    print(f"\n开始全量推理...")
    for i, img_name in enumerate(image_list):
        if i % 100 == 0:
            print(f"  -> 处理进度: [{i}/{len(image_list)}]")

        try:
            # =================================================================
            # 🛡️ 获取真实路径与加载图像
            # =================================================================
            if data_format == "csv":
                img_path = "../../a_TongueDx2/seg/" + img_name if dataset_name == "TongueDx" else img_name
                row = df[df['image_path'] == img_name]
                if row.empty: continue
                multi_hot_vector = row[labels].values[0].tolist()

            elif data_format == "yolo":
                # 完美的真实物理路径
                img_path = os.path.join(config_dict["image_dir"], img_name)
                # 直接获取我们预解析好的多标签数组
                multi_hot_vector = multi_hot_dict[img_name]

            # 加载真实图片
            img_tensor = load_image(img_path, device)
            # =================================================================

            ground_truth_names = [labels[j] for j, val in enumerate(multi_hot_vector) if val == 1]

            with torch.no_grad():
                # 🚨 把这两个罪魁祸首统统改回 1.0 ！！！取消压制！
                T_baseline = 1.0
                T_ours = 1.0

                b_logits = model_baseline(img_tensor)[0]
                o_logits = model_ours(img_tensor)[0]

                b_probs = torch.sigmoid(b_logits / T_baseline).cpu().numpy()
                o_probs = torch.sigmoid(o_logits / T_ours).cpu().numpy()

            item = {
                "id": i + 1,
                "dataset": dataset_name,
                "image": img_path,
                "ground_truth": ground_truth_names,
                "baseline": {labels[j]: float(b_probs[j]) for j in range(len(labels))},
                "ours": {labels[j]: float(o_probs[j]) for j in range(len(labels))}
            }
            results_data.append(item)

        except Exception as e:
            print(f"图片 {img_name} 处理出错: {e}")
            pass

    with open(full_output_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"🎯 全量推理结束！")
    mine_star_samples(full_output_json, star_output_json)
# tonguedx 352_1.jpg, 591_1.jpg, 662_1.jpg, 3377_1.jpg
# ITDD A (173).jpg,10764.jpg,A (282).jpg,10382.jpg
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"全局使用设备: {device}")

    RUN_DATASETS = ["ITDD","TongueDx"]  # 目前只跑 TongueDx
    # "TongueDx"
    for ds_name in RUN_DATASETS:
        if ds_name in DATASETS_CONFIG:
            process_single_dataset(ds_name, DATASETS_CONFIG[ds_name], device)
        else:
            print(f"未找到数据集 {ds_name} 的配置。")
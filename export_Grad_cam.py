import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 引入你的模型构建函数
from tonguedx_MLC.our_version7.paper_ablation_MTI_HANet import get_paper_ablation_model
from tonguedx_MLC.our_version7.optuna_train import _build_safe_config

# 引入 CTransCNN
from tonguedx_MLC.other_model.CTransCNN.CTransCNN import CTransCNN

# 设置中文字体（确保图表里的文字能正常显示）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 🗂️ 你的原始数据集配置 (绝不乱改)
# =====================================================================
DATASETS_CONFIG = {
    "TongueDx": {
        "labels": [
            'TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
            'Crack', 'Toothmark', 'FurThick', 'FurYellow'
        ],
        "json_file": "aaa_star_samples_tonguedx.json",
        "baseline_weights": "../our_version7/paper_ablation_best_model_paper_model_1_TongueDx_csra_seed3_e30_20260410_200836.pth",
        "baseline_json": "../our_version7/paper_ablation_best_params_model_1_TongueDx_20260410_200836.json",
        "ours_weights": "../our_version7/paper_ablation_best_model_mode4_ours_TongueDx_seed3_e30_20260422_163433.pth",
        "ours_json": "../our_version7/paper_ablation_best_params_mode4_ours_TongueDx_20260422_163433.json",
        "sota_weights": "../other_model/model_pth/fftrans_tonguedx.pth",
        "ctranscnn_weights": "../other_model/model_pth/ctranscnn_tonguedx.pth",
    },
    "ITDD": {
        "labels": [
            'HealthyTongue', 'PeelingCoating', 'RedTongue', 'PurpleTongue', 'ChubbyTongue',
            'ThinTongue', 'RedDot', 'Crack', 'Toothmark', 'FurWhite', 'FurYellow',
            'FurBlack', 'SmoothCoating'
        ],
        "json_file": "aab_star_samples_itdd.json",
        "baseline_weights": "../our_version7/paper_ablation_best_model_paper_model_1_ITDD_csra_seed3_e30_20260410_200836.pth",
        "baseline_json": "../our_version7/paper_ablation_best_params_model_1_ITDD_20260410_200836.json",
        "ours_weights": "../our_version7/paper_ablation_best_model_mode4_ours_ITDD_seed3_e30_20260422_205455.pth",
        "ours_json": "../our_version7/paper_ablation_best_params_mode4_ours_ITDD_20260422_205455.json",
        "sota_weights": "../other_model/model_pth/fftrans_itdd.pth",
        "ctranscnn_weights": "../other_model/model_pth/ctranscnn_itdd.pth",
    }
}

# =====================================================================
# 🤖 你自己写的解析逻辑 (保留不动)
# =====================================================================
def auto_extract_tasks():
    tasks = []
    for dataset_name, config in DATASETS_CONFIG.items():
        json_path = config["json_file"]
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        top_4_samples = data[:8]

        for item in top_4_samples:
            img_path = item["image"]
            gt_labels = item.get("ground_truth", [])
            reasons = item.get("star_reasons", [])

            target_class = None
            for r in reasons:
                if "抢救" in r or "全胜" in r:
                    match = re.search(r'\[(.*?)\]', r)
                    if match:
                        target_class = match.group(1)
                        break

            if not target_class and gt_labels:
                target_class = gt_labels[0]

            if target_class:
                tasks.append({
                    "dataset": dataset_name,
                    "image_path": img_path,
                    "target_class": target_class,
                    "id": item["id"]
                })
    return tasks

def get_target_layer(model, model_type):
    """根据不同的模型架构动态选择特征提取层"""
    if model_type == "ctranscnn":
        return [model.MBR_layers[-1]]
    else:
        return [model.cnnbackbone.feature_extractor[-1]]

# =====================================================================
# 🚀 修好的模型加载逻辑 (兼容了 CTransCNN 的坑)
# =====================================================================
def load_model(dataset_name, model_type, device):
    cfg_dict = DATASETS_CONFIG[dataset_name]
    labels = cfg_dict["labels"]
    num_classes = len(labels)
    weights_path = cfg_dict.get(f"{model_type}_weights")

    if not weights_path or not os.path.exists(weights_path):
        return None

    if model_type == "ctranscnn":
        model = CTransCNN(
            label_dim=512,
            cnn_channels=2048,
            transformer_dim=512,
            target_size=(7, 7),
            num_labels=num_classes
        )
    else:
        json_path = cfg_dict[f"{model_type}_json"]
        model_name_str = "model_1" if model_type == "baseline" else "ours"
        with open(json_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        cfg = _build_safe_config(params, batch_size=1, device=device, num_classes=num_classes)
        model = get_paper_ablation_model(model_name_str, labels, cfg)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

    if model_type == "ctranscnn":
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("t2c."):
                new_state_dict[k.replace("t2c.", "t2c_layers.0.")] = v
            elif k.startswith("c2t."):
                new_state_dict[k.replace("c2t.", "c2t_layers.0.")] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)

    model.to(device).eval()
    return model

# =====================================================================
# 🚀 核心流水线
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 检测到设备: {device} (4090 准备就绪)")

    print("🧠 正在全自动解析 JSON 文件并提取目标...")
    visual_tasks = auto_extract_tasks()
    print(f"🎯 成功锁定 {len(visual_tasks)} 个王牌样本准备出图！\n")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for idx, task in enumerate(visual_tasks):
        dataset_name = task["dataset"]
        img_path = task["image_path"]
        target_class_name = task["target_class"]
        sample_id = task["id"]

        print(f"{'=' * 60}")
        print(f"🚀 开始渲染 [第 {idx + 1}/{len(visual_tasks)} 张图] (样本ID: {sample_id})")
        print(f"📁 数据集: {dataset_name} | 🎯 核心展示特征: {target_class_name}")

        labels = DATASETS_CONFIG[dataset_name]["labels"]
        if target_class_name not in labels:
            print(f"❌ 错误: {target_class_name} 不在标签列表，跳过...")
            continue
        target_idx = labels.index(target_class_name)

        if not os.path.exists(img_path):
            print(f"❌ 找不到图片 {img_path}，请检查路径！跳过...")
            continue

        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img_float = np.float32(rgb_img) / 255

        tensor_img = transforms.ToTensor()(rgb_img)
        input_tensor = normalize(tensor_img).unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(target_idx)]

        # 1. Baseline
        model_baseline = load_model(dataset_name, "baseline", device)
        target_layers_b = get_target_layer(model_baseline, "baseline")
        with GradCAM(model=model_baseline, target_layers=target_layers_b) as cam:
            grayscale_cam_b = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image_b = show_cam_on_image(rgb_img_float, grayscale_cam_b, use_rgb=True)
        del model_baseline

        # 2. CTransCNN
        cam_image_c = np.zeros_like(rgb_img)
        model_ctrans = load_model(dataset_name, "ctranscnn", device)
        if model_ctrans is not None:
            target_layers_c = get_target_layer(model_ctrans, "ctranscnn")
            with GradCAM(model=model_ctrans, target_layers=target_layers_c) as cam:
                grayscale_cam_c = cam(input_tensor=input_tensor, targets=targets)[0, :]
                cam_image_c = show_cam_on_image(rgb_img_float, grayscale_cam_c, use_rgb=True)
            del model_ctrans

        # 3. Ours
        model_ours = load_model(dataset_name, "ours", device)
        target_layers_o = get_target_layer(model_ours, "ours")
        with GradCAM(model=model_ours, target_layers=target_layers_o) as cam:
            grayscale_cam_o = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image_o = show_cam_on_image(rgb_img_float, grayscale_cam_o, use_rgb=True)
        del model_ours

        # 4. 绘图保存
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        title_fontsize = 18

        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Input Image", fontsize=title_fontsize, pad=10)
        axes[0].axis('off')

        axes[1].imshow(cam_image_b)
        axes[1].set_title(f"Baseline (ResNet101)", fontsize=title_fontsize, pad=10)
        axes[1].axis('off')

        axes[2].imshow(cam_image_c)
        axes[2].set_title(f"CTransCNN", fontsize=title_fontsize, pad=10)
        axes[2].axis('off')

        axes[3].imshow(cam_image_o)
        axes[3].set_title(f"Ours (Proposed)", fontsize=title_fontsize, pad=10, fontweight='bold', color='darkred')
        axes[3].axis('off')

        plt.suptitle(f"Target Class: {target_class_name}", fontsize=22, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        save_path = f"GradCAM_4Col_{dataset_name}_ID{sample_id}_{target_class_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ 保存成功 -> {save_path}")

if __name__ == "__main__":
    main()
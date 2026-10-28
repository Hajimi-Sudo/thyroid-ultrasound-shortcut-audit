# 超越诊断准确性：深度学习甲状腺结节超声分析中捷径学习与非病灶混杂因子的临床审计

[![Language: English](https://img.shields.io/badge/Language-English-blue.svg)](README.md)
[![Language: 中文](https://img.shields.io/badge/Language-中文说明-green.svg)](README_zh.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Peer Review: Under Review](https://img.shields.io/badge/Peer%20Review-Under%20Review-blue.svg)]()
[![Domain: Medical AI & Sonography](https://img.shields.io/badge/Domain-Medical%20AI%20%26%20Sonography-0072B2.svg)]()
[![Status: Code Release](https://img.shields.io/badge/Status-Code%20Release-success.svg)]()

> **语言导航 / Language**: [English](README.md) | [中文说明](README_zh.md)

本开源代码仓库包含用于评估甲状腺结节超声深度学习系统捷径学习（Shortcut Learning）、非病灶伪影记忆与跨数据集泛化性能的官方评测套件、反事实扰动流程、多网络架构审计模型以及临床决策支持分析工具。

---

## 📌 核心发现与执行摘要

尽管深度学习模型在经过人工严格筛选的单中心甲状腺超声基准数据集上经常取得优异的诊断鉴别性能（$\text{AUROC} > 0.90$），但在真实临床应用中往往受到非病灶混杂因素的严重制约。本研究在卷积网络（ResNet-50、ConvNeXt-Tiny）与视觉 Transformer（Swin-Transformer Tiny）三大主流架构上开展了系统的临床与算法安全审计，核心发现如下：

1. **超声测量十字标记诱发预测劫持与敏感度崩溃**：
   - 在独立内部测试队列（$N = 623$）上反事实注入临床标准电子测量十字标记（$+$）会导致特征表征发生严重解构，诊断鉴别力从基线 $\text{AUROC } 0.812$（$95\%\text{ CI: } 0.778 - 0.844$）断崖式跌落至 $0.628$（$95\%\text{ CI: } 0.582 - 0.668$，成对 DeLong 检验 $p = 1.94 \times 10^{-18}$；成对 Wilcoxon 符号秩检验 $W = 1,420, p = 8.04 \times 10^{-14}$）。
   - 诊断敏感度由 $63.4\%$ 崩溃至 $9.4\%$（在 265 例恶性样本中假阴性误诊从 97 例暴增至 240 例），模型校准误差 $\text{ECE}$ 从 $0.149$ 恶化至 $0.223$。针对部分良性结节，测量标记引发了恶性预测概率的异常暴增（例如良性样本 Case 361 恶性风险从 $0.7\%$ 飙升至 $59.8\%$），导致良性结节被虚假推向侵入性穿刺决策。
2. **空间归因能量被非解剖标记劫持**：
   - Grad-CAM 特征归因量化分析表明，在全部 623 例测试样本中，模型有平均 $41.2\% \pm 5.8\%$（$95\%\text{ CI}: 40.7\% - 41.7\%$）的空间归因能量高度集中在仅 3 像素宽度的合成十字标记区域（良性结节平均 $41.8\% \pm 5.9\%$，恶性结节平均 $40.4\% \pm 5.6\%$），证实深度网络系统性地利用医师测量标记替代了结节内在病理特征（如低回声、微钙化或毛刺状边缘）。
3. **背景腺体纹理的虚假捷径记忆**：
   - 在将中心结节完全遮挡并用周围腺体实质纹理修补填充的探针条件（条件3：Inpainted Background）下，模型依然维持高达 $0.814$（$95\%\text{ CI}: 0.781 - 0.846$）的 $\text{AUROC}$、 $0.754$ 的 $\text{AUPRC}$ 以及 $81.8\%$ 的高特异度。这证明常规未受约束的深度网络能够仅凭病灶外的腺体背景纹理与厂商声学界面便达到甚至超过未遮挡全图的鉴别力（$0.814$ 对比 $0.812$），存在严重的背景信息泄漏依赖。
4. **病灶解剖 ROI 约束实现临床安全防护**：
   - 采用外扩 $10\%$ 解剖过渡边界的病灶感兴趣区域（ROI）约束训练与推理，能够彻底切断厂商外周界面与远端背景实质捷径。在独立的外部 DDTI 验证队列（$N = 347$ 幅图像 / 299 名患者）上，ROI 防护模型将图像级 $\text{AUROC}$ 恢复至 $0.751$（全图基线为 $0.654$，$p = 3.1 \times 10^{-4}$），患者级 $\text{AUROC}$ 恢复至 $0.758$（全图基线为 $0.662$，$p = 4.3 \times 10^{-4}$）。
   - 在锁定保证 $\text{敏感度} \ge 95.0\%$ 的临床高安全工作点下，ROI 防护模型在外部验证队列中实现了图像级 $51.1\%$ 和患者级 $52.7\%$ 的良性结节穿刺避免率（每 1,000 例良性患者避免 527 次非必要穿刺，显著优于全图基线的 246 次；在内部测试集避免 $58.7\%$ / 587 次），在确保恶性肿瘤极低漏诊的前提下显著减轻过度诊疗负担。

---

## 🔬 方法学实验审计框架

本研究系统构建并审计了五种受控输入表征条件：

| 审计条件 | 输入表征定义 | 实验目的与审计靶标 |
| :--- | :--- | :--- |
| **条件 1：原始全图（Raw Full-Frame）** | 未经裁切的原始 B 模超声全图（$224 \times 224$） | 评估标准“端到端”输入下模型对外周厂商状态栏、图头信息与背景纹理的未经约束易感性。 |
| **条件 2：清理全图（Cleaned Full-Frame）** | 裁除外周边框（左右及底边 $5\%$，顶边 $8\%$） | 检验单纯移除外周黑边与文字刻度能否消除非病灶偏见并维持声学背景。 |
| **条件 3：背景修补（Inpainted Background）** | 病灶区域完全遮挡并以周围正常实质纹理平滑填充 | 量化捷径容量：在完全无法观察病灶本身时测定模型的虚假诊断鉴别能力。 |
| **条件 4：十字加压（Caliper-Stressed）** | 在测试集图像结节长短轴合成注入临床测量十字标记 | 反事实压力测试：检验超声医师测量标记是否会诱发模型表征崩溃与虚假高恶性度报警。 |
| **条件 5：病灶 ROI 约束（Nodule ROI-Constrained）** | 沿结节边界外扩 $10\%$ 解剖边缘裁切病灶区域 | 强迫模型特征提取严格聚焦于病灶实质及即刻交界面，隔绝外部非病灶混杂因素。 |

---

## 📊 数据集架构与角色定义

本研究整合了三个标准化、经细胞学/组织病理学金标准确诊的公开甲状腺超声队列：

1. **主要开发与内部审计队列（Curated ThyroidXL）**：
   - 包含来自越南国家内分泌医院（日立 Aloka Arietta V70 设备）的 $N = 3,115$ 例患者确诊超声检查（$1,905$ 例良性，$1,210$ 例恶性），系自原始 4,093 例数据库经过严格的四阶段纳入排除审计后确立。
   - 在患者水平通过分层随机哈希严格划分为包含 $2,492$ 例患者的训练集（良性 $1,547$ 例，恶性 $945$ 例）与包含 $623$ 例患者的独立内部测试集（良性 $358$ 例，恶性 $265$ 例）。
   - 两集合交集严格为空（$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset$），彻底杜绝患者级跨集合数据泄漏。
2. **外部独立验证与真实游标压力队列（DDTI）**：
   - 包含来自哥伦比亚圣何塞医院（东芝超声设备）的 $N = 347$ 幅超声图像，涵盖 $299$ 名独立患者。
   - 具有 100% 真实的超声医师临床测量十字标记及放射科专家 ACR TI-RADS 分级（TR1–TR5）。
   - 同时采用图像级分析（$N = 347$，基于患者聚类 Bootstrap 计算 95% 置信区间）与患者级聚合分析（$N = 299$，均值池化与最大值池化敏感度分析）。
3. **跨设备多中心泛化评估队列（TN5000）**：
   - 包含来自中国医学科学院肿瘤医院（GE Logiq E9 与 Logiq S7）的 $N = 5,000$ 幅图像。

---

## 📁 仓库文件目录结构

```
.
├── README.md                      # 项目英文说明文档
├── README_zh.md                   # 项目中文说明文档
├── AGENTS.md                      # 研究规范与学术可复现性准则
├── requirements.txt               # Python 核心依赖环境列表
├── run_audit.sh                   # 自动化全流程审计执行脚本
├── .gitignore                     # Git 忽略文件规则
├── assets/
│   └── figures/                   # 高分辨率临床流程与实验结果图表
│       ├── figure1.jpg            # 研究工作流与审计框架总览图
│       ├── figure2.jpg            # 五大条件 ROC 诊断鉴别曲线图
│       ├── figure3.jpg            # Grad-CAM 归因与十字标记表征劫持可视化图
│       └── figure4.jpg            # 决策曲线分析（DCA）净获益曲线图
├── code/                          # 核心算法、模型与评估套件
│   ├── dataset_manifest.py        # 数据集元数据映射、验证与完整性校验
│   ├── mask_and_perturb.py        # ROI 提取、实质平滑修补与十字标记合成
│   ├── train_audit_models.py      # 多主干网络训练、消融条件与压力测试
│   ├── evaluate_metrics.py        # AUROC、ECE、DCA 净获益、Bootstrap CI 与归因能量计算
│   └── run_pilot_audit.py         # 快速端到端冒烟集成测试脚本
├── analysis/                      # 实验日志、指标汇总与 DCA 导出数据
│   └── experiment_results.json    # 五种表征条件的评估指标与净获益输出
└── docs/                          # 详细临床协议与研究技术文档
    ├── RESEARCH_BRIEF.md          # 科学研究设计与临床终点说明
    ├── PROBLEM_ANCHOR.md          # 捷径分类学与临床安全危害论证
    ├── DATASET_AUDIT.md           # 队列纳入排除与防泄漏审计协议
    └── EXPERIMENT_PROTOCOL.md    # 多条件消融与 DCA 验证协议
```

---

## 🚀 安装与快速上手

### 1. 环境准备

克隆本代码仓库并安装相关 Python 依赖库：

```bash
git clone https://github.com/Hajimi-Sudo/thyroid-ultrasound-shortcut-audit.git
cd thyroid-ultrasound-shortcut-audit
pip install -r requirements.txt
```

### 2. 管道完整性校验（冒烟测试）

运行轻量级集成脚本，验证 PyTorch GPU 加速与反事实扰动流水线：

```bash
python3 code/run_pilot_audit.py
```

### 3. 执行多条件全量审计

在不同骨干网络上对所有实验表征条件进行训练与评测：

```bash
# ResNet-50 基线审计
python3 code/train_audit_models.py --backbone resnet50 --condition all --epochs 8 --output_dir analysis

# ConvNeXt-Tiny 纯卷积模型审计
python3 code/train_audit_models.py --backbone convnext_tiny --condition all --epochs 8 --output_dir analysis

# Swin-Transformer Tiny 视觉 Transformer 审计
python3 code/train_audit_models.py --backbone swin_t --condition all --epochs 8 --output_dir analysis
```

或直接执行自动化脚本一键运行全流程：

```bash
bash run_audit.sh
```

---

## 📈 核心评估指标实现说明 (`code/evaluate_metrics.py`)

- **诊断准确性与校准度**：
  - `compute_diagnostic_metrics(y_true, y_prob, threshold)`：计算 AUROC、AUPRC、敏感度、特异度、阳性预测值（PPV）、阴性预测值（NPV）、Brier 分数以及 10 等宽分箱的期望校准误差（ECE）。
- **临床决策实用性（决策曲线分析）**：
  - `compute_decision_curve(y_true, y_prob, thresholds)`：在临床细针穿刺阈值范围（$p_t \in [0.05, 0.50]$）内计算模型辅助分流相比于“全穿刺”和“不穿刺”的净临床获益（Net Benefit）。
- **归因忠实度与十字标记能量分析**：
  - `compute_caliper_attribution_energy(saliency_map, caliper_mask)`：量化 Grad-CAM 显著性热力图中落在膨胀标记掩膜内的归因能量比例 $E_{\text{caliper}}$。
  - `compute_pointing_game_hit_rate(saliency_maps, masks)`：计算热力图最高激活点是否精准落在病灶真实轮廓内的命中率。
- **临床安全阈值标定**：
  - `calibrate_locked_high_sensitivity_threshold(y_true, y_prob, target_sensitivity=0.950)`：根据预设的高敏感度安全标准（$\ge 95.0\%$）在训练交叉验证中锁定操作截断值 $p^*$。
- **患者级聚类 Bootstrap 检验**：
  - `compute_patient_clustered_bootstrap_ci(patient_ids, y_true, y_prob, n_bootstraps=2000)`：针对外部独立验证队列中单患者多图的相关性，进行严格的按患者聚类重抽样检验与置信区间估计。

---

## ⚖️ 开源许可与伦理声明

- **代码开源协议**：本项目依据 [MIT 许可证](LICENSE) 开源发布。
- **医学伦理合规**：本研究所有实验均基于公开可获取的匿名去标识化回顾性临床超声数据集完成，严格遵守《赫尔辛基宣言》、HIPAA 隐私准则及相关伦理审查标准。

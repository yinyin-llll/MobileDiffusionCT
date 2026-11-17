# MobileDiffusionCT

可独立训练、验证和测试的当前实验代码：**全普通卷积＋多阶段残差自适应映射＋原残差去噪器＋仅重建损失**。

模型由二维 CNN/ViT 条件编码器、多阶段三维特征映射、三维自编码器和潜空间残差去噪器组成。提供配置的参数量为 **17,536,467**。二维特征按名义视角映射到三维，并通过残差门控逐阶段融合。

本仓库保留现有模型计算与权重命名，收窄了模型/数据工厂和命令默认值。模型核心源码可通过 `docs/source_manifest.json` 中的 SHA256 核对。源文件中的版权与 GPLv3 声明已保留，见 `LICENSE` 和 `NOTICE`。

## 安装

已验证环境为 Python 3.10、PyTorch 2.3.1、torchvision 0.18.1。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

仅做 CPU 检查时，可先安装对应的 CPU 版 PyTorch，再安装依赖。Windows 激活命令为 `.venv\Scripts\activate`。

可选 LPIPS：`python -m pip install -r requirements-metrics.txt`。首次启用 LPIPS 会下载公开的 AlexNet 权重；算法自身从头训练不需要预训练权重。

## 不需要真实数据的运行检查

在仓库根目录运行：

```bash
python -m tools.check_architecture --out verification/architecture.json
python -m tools.smoke_test
```

第二条命令在临时目录生成合成 HDF5，实际执行训练、验证、保存权重、重新加载、采样、指标、FLOPs 和体数据导出。它使用32³/两步采样和恒定学习率缩短检查时间，正式配置保持128³/300步及原学习率计划。合成测试分数没有科研含义。架构检查还可增加 `--checkpoint /path/to/1000_net_G.pth`，严格检查已训练权重的兼容性。

## 数据格式

图像、体数据、训练权重和实验输出均需自行准备。仓库包含原实验的去标识病例目录名划分清单，训练/验证/测试各为 **196/56/28**。

```text
YOUR_DATA_ROOT/
  datasets_10/
    ct_xray_data.h5
  ...
```

每个 HDF5 包含：

| 字段 | 形状 / 含义 |
|---|---|
| `ct` | `[D,H,W]`，数值编码应与原实验一致 |
| `xrays` | `[2,H,W]`，浮点强度 `[0,1]` |
| `angles` | `[0,90]`，单位为度 |
| 属性 `input_type` | DRR 配置为 `orthogonal_ap_lateral_drr_from_uploaded`；真实配置为 `real_lidc_xray_2view` |

CT 采用三线性插值，X-ray 采用双线性插值，统一为128³和128²。当前实验直接将 CT 数值裁剪到 `[-2000,4000]` 后映射为 `[0,1]`；加载器不执行 DICOM 到 HU 的转换或减1024。若使用其他数据，应先确认其数值编码。带有 `drr_display_orientation=flipud` 的文件会按原协议恢复显示翻转。

真实正侧位配置使用 `nominal_parallel` 几何；它依据0°/90°标注进行特征映射，没有逐病例相机标定。划分清单检查病例 ID；患者级防泄漏需要数据提供方另行核对患者映射。

## 训练 DRR 正侧位 2V

```bash
python train.py --dataroot /path/to/DRR_APLAT_2V \
  --ymlpath configs/standard_aplat_2v.yml \
  --datasetfile splits/DRR_APLAT_2V/train.txt \
  --valid_datasetfile splits/DRR_APLAT_2V/val.txt \
  --valid_dataset val --tag mobilect_standard_aplat2v --gpu 0 --seed 2026
```

默认从头训练1000轮。重建损失为体素 L1，权重2.6；其他损失权重为0。配置使用 ViT、一层四头注意力、多阶段自适应映射、全普通卷积和原残差去噪器，`diffusion_condition_detail: none`。训练使用1000步余弦噪声调度，测试采用300步确定性 DDIM、CFG1.8、EMA0.9997。

更换新实验时使用新的 `--tag`，避免覆盖已有训练目录。正式训练使用 CUDA；`--gpu ""` 可切换 CPU，适合运行合成检查。

## 测试和推理

```bash
python test.py --dataroot /path/to/DRR_APLAT_2V \
  --ymlpath configs/standard_aplat_2v.yml \
  --datasetfile splits/DRR_APLAT_2V/test.txt \
  --tag mobilect_standard_aplat2v --check_point 1000 --gpu 0 --seed 42 \
  --save-volumes --profile-flops --inference-warmup --resultdir results/aplat2v
```

增加 `--lpips --lpips-device cpu` 可输出 LPIPS。测试需要已训练权重；默认根据相同 `--tag` 和轮数定位它。原生模型的 `--load_path` 接受 checkpoint 根目录，而非单个 `.pth` 文件。

主要输出：

```text
save_models/multiView_CTDiffusion/LIDC256/TAG/checkpoint/EPOCH/EPOCH_net_G.pth
save_models/multiView_CTDiffusion/LIDC256/TAG/train_log/  # TensorBoard
results/aplat2v/LIDC256/test_1000/metrics_per_case.csv
results/aplat2v/LIDC256/test_1000/metrics_summary_wide.csv
results/aplat2v/LIDC256/test_1000/performance.json
results/aplat2v/LIDC256/test_1000/flops.json
results/aplat2v/LIDC256/test_1000/volumes/
```

`tensorboard --logdir save_models --port 6006` 可查看训练 loss。测试报告均值、病例标准差和95%置信区间半宽。FLOPs 按完整重建计算，1 MAC=2 FLOPs；耗时同步 CUDA，排除数据读取和外部质量指标。共享 GPU 下的耗时应与独占测速区别记录。

## 真实配对正侧位 2V

将上述命令中的配置替换为 `configs/standard_real_2v.yml`，划分目录替换为 `splits/LIDC_REAL_XRAY_2V`，数据根目录替换为对应真实配对目录，并使用新标签（例如 `mobilect_standard_real2v`）。同一病例目录应存储配对 CT 和两幅 X-ray，病例名需与所选清单一致。

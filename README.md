# YOLO-Seg 训前 DINOv2 数据审查框架

面向 YOLO-Seg 训练前数据质量审查的工程框架。项目遵循“单次前向推理、特征多路复用”的原则：先通过纯几何规则筛除物理不可学样本，再使用 `dinov2_vitb14_reg` 提取特征，并将特征复用于多类量化审查引擎，最终导出可供标注复核的可视化图集与修改 SOP。

整体方案详见：[YOLO-Seg 训前 DINOv2 数据审查框架.md](./YOLO-Seg%20训前%20DINOv2%20数据审查框架.md)。

## 审查流程

1. **Stage 0 - 几何解析与物理极限预检**：解析 YOLO-Seg 多边形，计算紧致 BBox、面积、长宽比、填充率，并模拟 Proto 1/4 下采样后的掩膜留存情况。
2. **Stage 1 - OK 记忆库与动态阈值标定**：将同名 LabelMe JSON 中没有有效 Polygon 的图片视为 OK，固定切分为互不泄露的 `OK_Bank` 与 `OK_Audit`，通过 Audit 对 Bank 的近邻距离分布标定正常和显著异常阈值。
3. **Stage 2 - DINOv2 双流特征提取**：
   - 实例级流：围绕标注实例裁剪，生成局部异常图、实例向量和几何向量。
   - 整图级流：滑窗扫描整图，生成全图异常热力图和正常区域特征池。
4. **量化审查与报告导出**：复用异常图、实例特征、几何特征和全图热力图，执行多项审查并输出图集。

## 五类输出图集

| 图集 | 目标问题 | 主要判定依据 |
| --- | --- | --- |
| 图集 1：疑似过标 / 虚标 | 标注内容与正常背景无明显差异 | Mask 内异常弱、特征空间邻近 OK |
| 图集 2：灰区对齐与拆类 | 类间边界模糊、单类内部形态混杂 | 线性探针灰区、UMAP 和 HDBSCAN 子聚类 |
| 图集 3：粗框 / 一框包多点 / 错位 | Polygon 包含过多背景或边界错位 | 内部背景污染率、内外环带异常落差 |
| 图集 4：疑似漏标扫雷 | 标注区域外存在显著异常区域 | 屏蔽已标注区域后的异常连通域检测 |
| 图集 5：Proto 1/4 消失 / 断裂预警 | 小目标在分割原型尺寸下不可学习 | 下采样 IoU、非零像素数、连通域断裂 |

## 目录结构

```text
.
├── configs/              # 模型、阈值与运行配置
├── data/
│   ├── raw/              # 原始 YOLO-Seg 数据集
│   ├── interim/          # 中间切片、临时掩膜和缓存
│   ├── processed/        # 清洗或标准化后的数据
│   └── ok_bank/          # OK 记忆库与 Faiss 索引产物
├── docs/
│   ├── architecture/     # 架构设计和技术说明
│   └── sop/              # 标注复核和数据清洗 SOP
├── outputs/
│   ├── artifacts/        # 特征、索引、聚类和统计产物
│   ├── logs/             # 运行日志
│   └── reports/          # PNG、HTML 和汇总审查报告
├── scripts/              # 命令行和 CI 执行入口
├── src/
│   ├── calibration/      # OK_Bank、Faiss 与动态阈值标定
│   ├── engines/          # Mask 纯度、漏标扫描、聚类和线性探针引擎
│   ├── features/         # DINOv2 模型加载、切片和双流特征提取
│   ├── geometry/         # 多边形、BBox 与 Proto 1/4 几何预检
│   ├── pipeline/         # 端到端流程编排
│   ├── reports/          # 五类图集及 HTML/PNG 渲染
│   └── utils/            # 配置、日志、IO 和可视化公共能力
└── tests/
    ├── integration/      # 端到端流程集成测试
    └── unit/             # 模块级单元测试
```

## Stage 0：832 / 208 几何与 Proto 预检

Stage 0 固定使用 YOLOv11-Seg 训练尺寸 `832 x 832`，并按 `1/4` 分割原型步长模拟 `208 x 208` Proto 网格。实现遵循等比例 Letterbox 映射：先将原图与 Polygon Mask 对齐到 `832 x 832`，再下采样、二值化并还原，以计算 `IoU_proto`、Proto 非零像素数和连通域断裂。

人工审查结果以 HTML 图集为主：文字、指标与图表采用浏览器原生文字和 SVG 矢量图，原图与 Mask 面板保留 PNG；机器可读中间结果为 JSON，项目不生成 CSV 结果文件。

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
python scripts/run_stage0.py --input-dir data
```

默认输入优先支持当前数据目录的 LabelMe `PNG + JSON` 标注，也兼容标准 YOLO-Seg `TXT + 图片` 标注。报告默认写入 `outputs/reports/stage0/`，指标与实例记录写入 `outputs/artifacts/stage0/`。

## Stage 1：OK 记忆库与动态阈值标定

Stage 1 使用官方 `dinov2_vitb14_reg` 在 `448 x 448` 固定物理视野上提取 L2 标准化特征。无有效 Polygon 的图片划入 OK 候选集，并按固定随机种子切为 `80% OK_Bank / 20% OK_Audit`。Audit 从不进入 Bank，阈值由 Audit 对 Bank 的 1-NN 余弦距离分布自动标定：`P95` 为预警线，`P99.5 x 1.2` 为显著异常线。缺陷实例查询使用以实例中心为锚点的固定 `448 x 448` 上下文 Tile，确保 Query、Bank 与 Audit 的物理视野一致。

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
python scripts/run_stage1.py --input-dir data --top-n 100
```

报告写入 `outputs/reports/stage1/`，包含缺陷 Query 与最近 OK Tile 的可读对照；产物写入 `outputs/artifacts/stage1/`，包括阈值、查询记录、OK Tile 元数据及压缩特征矩阵（NPZ）。

## Stage 2：DINOv2 Patch 双流异常审查

Stage 2 复用 `dinov2_vitb14_reg` 的 `32 x 32` Patch Token。由隔离 OK_Bank 聚合同位置正常 Patch 原型，并允许 Query 在相邻 `2` 个 Patch 的位置半径内寻找最相似正常原型；Patch 阈值只由独立 OK_Audit 标定。实例流评估 Polygon 内外的异常分布，用于发现疑似虚标、粗标和边界错位；整图流通过重叠滑窗的最小异常距离融合，屏蔽现有标注后提取疑似漏标，同时审查 Audit OK 图是否存在污染。

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
python scripts/run_stage2.py --input-dir data --top-n 100
```

报告写入 `outputs/reports/stage2/`，包括疑似虚标、粗标/错位、疑似漏标和 OK 污染四类 HTML 图集；机器可读的阈值、实例指标、候选清单和 Patch 原型写入 `outputs/artifacts/stage2/`。不保存无必要的全量高分辨率异常图，避免审查产物膨胀。

## 五大指导图集生成

完成 Stage 0–2 后执行：

```bash
python -m pip install -r configs/guidance-requirements.txt
python scripts/run_guidance.py --top-n 100
```

统一入口为 `outputs/reports/guidance/index.html`，包含过标/虚标三联图、灰区概率梯度与逐类 UMAP/HDBSCAN 九宫格、粗标污染高亮图、漏标全图定位及放大热图、Proto 消失/断裂对照，以及人工复核 SOP。完整指标、折外预测、分组隔离证据、子簇信息和实例 CLS 特征写入 `outputs/artifacts/guidance/`；不生成 CSV，也不改写标注。

灰区探针使用按原图 SHA-256 分组的五折逻辑回归，所有同图实例及精确重复图像均留在同一折。概率为未校准模型分数，0.40–0.60 为核心灰区，0.35–0.65 为展示上下文。形态子簇使用 CLS 与 0.3 倍标准化几何特征；UMAP 用于展示，HDBSCAN 在组合特征空间聚类。少于 10 个实例的类别跳过聚类。图集 4 复用 Stage 2 的候选，OK 图审查范围为 Audit，按图像哈希和候选位置归并重复项。

当前 Stage 1 未保存实例 CLS，Stage 2 未保存用于图集的异常图，因此首次生成会补算实例 CLS/Patch 和候选局部热图；不重建正常库。单次前向同时复用 CLS 和 Patch。所有热图使用统一的高阈值色标，局部热图为中心上下文重算结果，与滑窗融合分数分别解释。

## 数据约定

- 输入数据为原图与 YOLO-Seg `.txt` 标注；每行标注包含类别与归一化多边形顶点。
- `data/raw/` 保持源数据只读；所有中间结果写入 `data/interim/` 或 `outputs/artifacts/`。
- 经人工复核确认的正常样本用于构建 `data/ok_bank/` 中的 OK 记忆库。
- 争议标注不得仅删除 `.txt` 中的多边形行后保留原图；应移除整图，或对争议区域进行合规的背景修复后再参与训练。

## 推荐实现顺序

1. 完成 YOLO-Seg 标注读取、多边形栅格化、紧致 BBox 和 Proto 1/4 模拟。
2. 完成 DINOv2 推理封装、切片策略、Patch 特征与异常距离计算。
3. 构建 OK 记忆库、Faiss 检索和动态阈值标定流程。
4. 实现四个审查引擎及严重度排序。
5. 实现五类 PNG/HTML 图集、汇总统计和标注修改 SOP 导出。
6. 将完整流程封装为 `scripts/` 下可供 CI 调用的命令行入口。

## 设计原则

- 单次 DINOv2 前向推理，最大化复用实例级与整图级特征。
- 阈值由 OK 审计集的实际距离分布标定，避免静态经验参数。
- 输出面向标注复核：每一条告警均应带有原图、热力图、对照样本或明确的几何证据。
- 保留中间产物的可追溯性，支持训练前数据门禁和问题回放。

# Stage 0：几何解析与 Proto 可学习性预检

## 1. 目标与边界

Stage 0 用于判断标注实例在 YOLOv11-Seg 的训练表示下是否具备基础可学习性，而不是判断缺陷是否真实存在或标注语义是否正确。

统一训练尺寸为 `832 x 832`。按分割 Proto 的 `1/4` 步长，统一模拟 `208 x 208` 的 Proto 网格。

本阶段主要识别三类问题：

1. 标注几何无效，例如顶点不足、坐标越界或多边形自交。
2. 目标过小或过细，映射至 Proto 网格后几乎消失。
3. 目标在低分辨率 Proto 网格中发生断裂，难以稳定学习其连续形态。

Stage 0 不能识别虚标、漏标、类别混杂或外观相似的灰区。这些问题应由后续 DINOv2 特征阶段处理。

## 2. 输入与适配

当前默认输入格式为 LabelMe：

- 原图：`PNG`、`JPG`、`JPEG`、`BMP` 或 `WEBP`
- 标注：与原图同名的 LabelMe `JSON`
- 标注形状：`polygon`

同时保留标准 YOLO-Seg 输入的兼容能力：

- 原图：与 TXT 同名的图片文件
- 标注：YOLO-Seg `.txt`，每行包含类别 ID 和归一化多边形坐标

当前实际数据集包含 462 组 `PNG + JSON` 文件，共 439 个 Polygon 实例，覆盖 8 个缺陷类别。

## 3. 实现流程

### 3.1 标注解析与几何检查

对每个 Polygon 执行以下检查与计算：

- 顶点数量是否不少于 3。
- 坐标是否为有限数值。
- 顶点是否越出原图边界。
- 多边形是否自交。
- Polygon 的紧致 BBox。
- Mask 面积、BBox 面积与填充率。
- BBox 长宽比。
- Polygon 凸度。

其中，紧致 BBox 是由 Polygon 的极值坐标推导而来，用于后续局部裁剪与几何质量检查。

### 3.2 YOLOv11 Letterbox 映射

原图和 Polygon Mask 按等比例缩放并居中填充至 `832 x 832`。

该步骤不直接拉伸原图为正方形，而是模拟 YOLO 系列常用的 Letterbox 输入方式。这样可以保留原目标的形状比例，并真实反映训练画布中的目标尺寸。

### 3.3 Proto 1/4 模拟

在 `832 x 832` Mask 的基础上执行：

```text
原始 Mask
  -> Letterbox 后的 832 x 832 Mask
  -> 面积插值缩放至 208 x 208 Proto 网格
  -> 0.5 阈值二值化
  -> 最近邻还原至 832 x 832
  -> 与原始训练画布 Mask 对比
```

此过程用于量化低分辨率 Proto 对目标轮廓、面积和连通性的保留能力。

### 3.4 风险判定

默认规则如下：

| 规则 | 阈值 | 含义 |
| --- | ---: | --- |
| Proto 留存率 | `IoU_proto < 0.35` | 下采样后轮廓与原始标注差异过大 |
| Proto 有效像素 | `< 4` 像素 | 目标在 `208 x 208` 网格中几乎消失 |
| 连通性 | 单连通变多连通 | 细线或狭窄区域在 Proto 网格中断裂 |
| 几何合法性 | 任一非法项 | 顶点、越界、自交等问题需直接修复 |
| 类别级预警 | 高风险占比 `> 15%` | 类别可能需要调整采图、切图或标注 SOP |

风险等级为：

- `critical`：几何无效，或同时触发关键 Proto 风险。
- `high`：目标几乎消失、显著断裂或 Proto 留存严重不足。
- `medium`：存在轻度留存或几何风险，建议复核。
- `pass`：未触发当前 Stage 0 风险规则。

## 4. 核心指标说明

### 4.1 `IoU_proto`

`IoU_proto` 是原始训练画布 Mask 与 Proto 下采样后还原 Mask 的交并比。

- 越接近 `1`，表示 Proto 网格能够较好保留原始轮廓。
- 低于 `0.35`，表示轮廓、面积或边缘细节丢失严重。

该指标反映形状留存，但不能单独代表可学习性。

### 4.2 `proto_nonzero_pixels`

该指标表示实例在 `208 x 208` Proto 网格中最终占据的非零像素数量。

- 少于 4 个像素时，目标即使仍有一定 IoU，也缺乏稳定的面积、边界和形状信息。
- 训练中此类目标的分割梯度通常不稳定，容易造成漏检或边界模糊。

因此，`IoU_proto` 与 `proto_nonzero_pixels` 必须联合判断。

### 4.3 连通域变化

对下采样前后的 Mask 统计连通域数量：

```text
original_components -> restored_components
```

如果原始目标为单连通区域、还原后变为多个连通区域，说明连续轮廓已经在 Proto 网格中断裂。该现象常见于细长划痕、窄边溢镀、狭窄缺口和点状目标。

### 4.4 类别高风险比例

类别高风险比例定义为：

```text
(critical 实例数 + high 实例数) / 该类别总实例数
```

当比例高于 15% 时，问题通常不是单个异常样本，而可能涉及：

- 缺陷类别本身过细、过小。
- 原始采图分辨率或视野设置不足。
- 训练前未采用切图策略。
- 标注外扩宽度不足。
- 类别定义需要更粗粒度的合并或拆分。

## 5. 当前数据集结果

以下结果来自当前 `data/` 目录的实际运行：

| 指标 | 结果 | 含义 |
| --- | ---: | --- |
| 标注文件数 | 462 | 当前审查覆盖的标注文件数 |
| 含实例图片数 | 235 | 至少包含一个 Polygon 的图片数 |
| Polygon 实例数 | 439 | 实际进入 Stage 0 计算的实例数 |
| 通过 | 409 | 未触发当前几何或 Proto 风险规则 |
| 中风险 | 10 | 存在轻度风险，建议人工复核 |
| 高风险 | 19 | Proto 表达能力明显不足 |
| 严重 | 1 | 存在关键 Proto 风险，应优先处理 |
| 高风险以上 | 20 | 建议进入训练前人工复核清单 |

风险等级分布：

```text
critical: 1
high:     19
medium:   10
pass:     409
```

## 6. 类别级评估

| 类别 | 实例数 | 严重 | 高风险 | 中风险 | 通过 | 高风险占比 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 涂层划痕 | 6 | 0 | 0 | 0 | 6 | 0.0% | 未触发 Proto 风险 |
| 涂层异物 | 112 | 0 | 0 | 0 | 112 | 0.0% | 未触发 Proto 风险 |
| 涂层漏铜 | 29 | 0 | 0 | 1 | 28 | 0.0% | 存在少量中风险实例 |
| 涂层破损 | 78 | 0 | 0 | 2 | 76 | 0.0% | 存在少量中风险实例 |
| 溢镀 | 75 | 0 | 16 | 6 | 53 | 21.3% | 超过 15% 策略预警线 |
| 电极漏铜 | 92 | 1 | 3 | 1 | 87 | 4.3% | 有个别严重小目标问题 |
| 电极缺损 | 2 | 0 | 0 | 0 | 2 | 0.0% | 样本量较少，暂未触发风险 |
| 磁体缺损 | 45 | 0 | 0 | 0 | 45 | 0.0% | 未触发 Proto 风险 |


人工审查以 HTML 为主：

- HTML 负责显示标题、指标、原因和建议，文字由浏览器原生渲染，可缩放且不会马赛克。
- SVG 负责总览统计图，文字与图表均为矢量对象。
- PNG 仅保留原图局部、Polygon/BBox 叠加和 Proto Mask 对比等图像内容。
- JSON 仅用于后续 Stage 1/2 消费、审计和结果追溯。

主要输出：

```text
outputs/reports/stage0/
├── index.html                              # Stage 0 总览和全部风险样本详情
├── overview.svg                            # 矢量总览统计图
├── proto_warning_gallery.html              # Proto 风险样本图集
├── proto_warning_contact_sheet_01.html     # 带文字的风险联系表
├── proto_warning_contact_sheet_02.html
├── proto_warning_contact_sheet_03.html
└── samples/
    ├── *_original.png                      # 原图局部 + Polygon/BBox
    └── *_proto.png                         # 832 Mask 与 Proto 还原对比

outputs/artifacts/stage0/
├── stage0_summary.json                     # 数据集级汇总
├── instances.json                          # 全部实例计算结果
└── flagged_instances.json                  # 非 pass 实例
```

## 8. 使用方法

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
python scripts/run_stage0.py --input-dir data
```

默认输出：

```text
outputs/reports/stage0/index.html
outputs/artifacts/stage0/stage0_summary.json
```

# Stage 2：分簇、多参考 Patch 异常审查技术报告

> 适用项目：YOLO-Seg 训前 DINOv2 数据审查框架  
> 特征模型：`dinov2_vitb14_reg`  
> 关联阶段：Stage 0 几何/Proto 预检、**新版 Stage 1 分簇 OK 记忆库与动态阈值标定**  
> 当前实现位置：`src/features/stage2_runner.py`

---

## 1. 阶段定位

Stage 2 的任务不是重新判断某个缺陷局部“整体像不像 OK”，而是在新版 Stage 1 已完成的场景分簇与正常区域对齐基础上，回答以下空间问题：

1. 已标注 Polygon 内是否存在稳定的视觉异常证据？
2. Polygon 是否包含大量与正常区域一致的背景，即疑似虚标、粗标或一框包多点？
3. Polygon 外环是否比 Polygon 内环更异常，即疑似标注边界错位或标注画小？
4. 已标注区域以外，是否存在值得人工检查的显著异常区域，即疑似漏标？
5. Stage 1 的 OK Audit 图中是否存在稳定异常峰值，即 OK 样本污染候选？

Stage 2 的所有输出均为**人工复核线索**，不会自动修改、删除或补充标注。

---

## 2. 与新版 Stage 1 的关系

### 2.1 职责边界

新版 Stage 1 负责建立正确的比较语义：

```text
原始图片
  → DINOv2 CLS 全图特征
  → OK 图像自适应聚类（场景/机台簇）
  → 同簇 OK_Bank 与 OK_Audit 隔离切分
  → 同簇、同归一化位置的多 OK 参考对齐
  → Tile 级动态阈值标定与缺陷实例整体距离
```

Stage 2 不重复聚类，也不再从全局 OK 图中搜索参考区域。它继承 Stage 1 的结果：

```text
缺陷实例
  → assigned_ok_cluster_id
  → normalized_center
  → 同簇全部 OK_Bank 图
  → Stage 1 median_ok_tile（仅用于展示）
```

因此两阶段的关系可简化为：

```text
Stage 1：找对照
Stage 2：解释差异在哪里
```

### 2.2 Stage 1 输出契约

Stage 2 读取：

```text
outputs/artifacts/stage1/
├── instance_queries.json
├── ok_bank_metadata.json
└── stage1_summary.json
```

每个 Stage 1 实例查询应包含至少以下字段：

```json
{
  "instance_id": "54:1",
  "image_path": "data/54.png",
  "assigned_ok_cluster_id": 0,
  "normalized_center": {"u": 0.76571429, "v": 0.914375},
  "bank_support": 29,
  "median_ok_tile": {
    "image_path": "data/1467.png",
    "x": 1461,
    "y": 1239,
    "width": 448,
    "height": 361
  }
}
```

其中：

- `assigned_ok_cluster_id`：缺陷图被归属到的 OK 场景/机台簇。
- `normalized_center`：缺陷中心在原图中的归一化坐标，用于跨尺寸图像同位置对齐。
- `bank_support`：该簇内可参与比较的 OK-Bank 图数量。
- `median_ok_tile`：Stage 1 按整体 Tile 距离排序后选择的中位数参考，**只用于报告展示**。

---

## 3. 为什么不能使用全局 OK Patch 原型

当前数据包含不同场景、不同机台和轻微物理位置偏差。若把所有 OK 图的 Patch 混合为全局正常库，会产生错误比较：

```text
NG 图中的电极 Patch
  ↔ 被拿去与另一机台、另一场景的背景 Patch 比较
  → 产生结构性假异常
```

因此 Stage 2 明确禁止以下旧逻辑：

```text
全部 OK 图 / 全部场景
  → 同位置 Patch 求均值
  → 全局 Patch 原型
  → 任意 NG Patch 与全局原型比较
```

当前使用的正确比较域是：

```text
同一个 Stage 1 场景簇
+ 同一个归一化空间位置
+ 全部 OK-Bank 参考图
```

即：

```text
NG Query Patch(r, c)
  ↔ { OK_i Patch(r, c) | OK_i 属于同簇，i = 1 ... K }
```

---

## 4. DINOv2 Patch 特征

### 4.1 输入与输出尺度

每个局部区域以固定物理视野输入：

```text
输入 Tile：448 × 448
模型：dinov2_vitb14_reg
Patch 网格：32 × 32
Patch 特征维度：768
```

每个 Patch 覆盖约：

```text
448 / 32 = 14 像素
```

模型输出：

```text
CLS Token：1 × 768
Patch Token：32 × 32 × 768
```

- CLS Token 在 Stage 1 负责全局 Tile 级聚类、对齐与动态阈值。
- Patch Token 在 Stage 2 负责局部异常定位与空间统计。

### 4.2 L2 标准化

每个 Patch Token 均进行 L2 标准化：

\[
\hat{z} = \frac{z}{\|z\|_2}
\]

使余弦相似度可以直接通过内积计算：

\[
\cos(q, r) = q^\top r
\]

---

## 5. 多参考平均 Patch 距离

设：

- \(q_{r,c}\)：缺陷 Query Tile 在 Patch 位置 \((r,c)\) 的 L2 标准化特征；
- \(r^{(i)}_{r,c}\)：同簇、同归一化位置的第 \(i\) 个 OK-Bank 参考特征；
- \(K\)：该簇 OK-Bank 图数。

Stage 2 的异常分数为：

\[
d_{r,c} = \frac{1}{K}\sum_{i=1}^{K}\left(1 - q_{r,c}^\top r^{(i)}_{r,c}\right)
\]

含义：

- 分数接近 `0`：Query Patch 与该簇正常参考整体一致；
- 分数较高：Query Patch 相对同簇正常模式存在明显偏离。

### 5.1 加速等价变换

上式可化简为：

\[
d_{r,c} = 1 - q_{r,c}^\top \left(\frac{1}{K}\sum_{i=1}^{K}r^{(i)}_{r,c}\right)
\]

因此可缓存：

\[
\bar{r}_{r,c} = \frac{1}{K}\sum_i r^{(i)}_{r,c}
\]

并计算：

\[
d_{r,c} = 1 - q_{r,c}^\top \bar{r}_{r,c}
\]

这与“逐个计算全部参考距离后取均值”严格等价，但避免每个 NG Query 与全部 OK 参考重复前向计算。

> 注意：这里缓存的均值特征不是旧版跨场景全局 Prototype。它的键是：`(Stage 1 cluster_id, normalized_center)`，只描述某一簇、某一精确对齐位置的多参考均值。

---

## 6. Patch 级动态阈值标定

### 6.1 为什么不能直接使用 Stage 1 阈值

Stage 1 阈值的度量对象是：

```text
448 × 448 Tile 的 CLS 向量距离
```

Stage 2 阈值的度量对象是：

```text
32 × 32 Patch Token 的逐位置多参考距离
```

两类距离分布不同，不能混用。

### 6.2 分簇 Audit 标定流程

对于每个 Stage 1 OK 簇：

```text
该簇 OK_Audit 图
  → 使用 Stage 1 对齐位置裁剪 Tile
  → 与同簇、同位置全部 OK-Bank 参考比较
  → 收集全部 Patch 多参考平均距离
  → 形成该簇的正常 Patch 分布
```

阈值规则：

```text
warning_threshold = P95(Audit Patch Distance)
high_threshold    = P99.5(Audit Patch Distance) × 1.2
```

当前正式运行的簇级阈值：

| OK 簇 | Patch 预警线 P95 | Patch 显著异常线 P99.5 × 1.2 |
|---:|---:|---:|
| 0 | 0.4744 | 0.7103 |
| 1 | 0.4930 | 0.7265 |

每个实例或整图滑窗仅使用其 `assigned_ok_cluster_id` 对应的阈值。

---

## 7. 流 A：实例级标注质量审查

### 7.1 输入流程

针对每个已有 Polygon 实例：

```text
Polygon / 紧致 BBox
  → 使用 Stage 1 normalized_center 截取 448 × 448 Query Tile
  → 获取同簇、同位置全部 OK-Bank Tile
  → DINOv2 Patch Token
  → 多参考平均 Patch 距离图：32 × 32
  → 双线性插值到 448 × 448 Anomaly Map
  → 将 Polygon 映射至 Query Tile
  → 计算 Mask 内外统计
```

### 7.2 输出指标

| 指标 | 含义 |
|---|---|
| `mask_mean_anomaly` | Polygon 内异常分数均值。越低越接近同簇 OK。 |
| `mask_normal_pixel_ratio` | Polygon 内低于本簇 `warning_threshold` 的像素比例。 |
| `bbox_background_mean_anomaly` | 紧致 BBox 内、Polygon 外区域的异常均值。 |
| `inner_ring_anomaly` | Polygon 内侧边缘环带的异常均值。 |
| `outer_ring_anomaly` | Polygon 外侧边缘环带的异常均值。 |
| `high_components` | Polygon 内高于本簇 `high_threshold` 的异常连通域数。 |

### 7.3 候选规则

#### 疑似虚标 / 视觉弱异常

```text
mask_mean_anomaly < cluster warning_threshold
AND
mask_normal_pixel_ratio >= 0.70
```

解释：在正确场景簇、正确空间位置、全部 OK 参考的对比下，Polygon 内大部分区域仍接近正常。

这表示“优先人工复核”，可能是：

- 虚标；
- 灰区；
- 业务标签定义边界不清；
- 缺陷真实存在但 DINO Patch 未能感知；
- 极小目标被上下文稀释。

#### 疑似粗标 / 一框包多点

```text
mask_normal_pixel_ratio >= 0.40
OR
high_components >= 2
```

解释：Polygon 内大量区域接近正常，或者内部存在多个分离的强异常区域。

#### 疑似边界错位

```text
outer_ring_anomaly >= inner_ring_anomaly + boundary_margin
```

解释：Polygon 外侧比内部更异常，可能存在：

- 多边形画小；
- 多边形整体偏移；
- 实际异常边缘未被覆盖；
- 邻近结构被错误标注。

#### 强视觉异常确认

```text
mask_mean_anomaly > cluster high_threshold
```

解释：Polygon 内具有很强的局部视觉异常证据。这不是标注错误告警，而是“异常证据强”的确认标记。

---

## 8. 流 B：整图疑似漏标与 OK 污染扫描

### 8.1 精确滑窗位置缓存

整图流使用：

```text
Tile size = 448
Stride = 358
Overlap ≈ 20%
```

数据中主要存在两种图像尺寸：

```text
2200 × 1600
2100 × 1600
```

这些尺寸对应实际出现的 `60` 个归一化滑窗中心。

为兼顾精确对齐与速度：

```text
每个 (cluster_id, exact_normalized_tile_center)
  → 对该簇全部 OK-Bank 图裁剪同一位置
  → 提取/聚合平均 Patch 特征
  → 缓存

NG / Audit 图滑窗
  → 仅一次 DINOv2 前向
  → 与相同 (cluster_id, exact_center) 的缓存参考比较
```

这里不存在“最近 3×3 网格”或周围位置近似匹配；使用的是数据中实际滑窗位置的精确键。

### 8.2 重叠 Tile 融合

同一像素可能被多个滑窗覆盖。融合使用：

```text
minimum anomaly distance
```

即：

```text
A(x, y) = min(A_tile_1(x, y), A_tile_2(x, y), ...)
```

理由：若某像素在任一上下文 Tile 中能够被正常参考较好解释，就不应因另一个 Tile 的边界上下文而被高分误报。

### 8.3 疑似漏标候选

对 NG 图整图异常图：

```text
整图异常图
  → 屏蔽已有 Polygon
  → Polygon 外扩 5 px 安全区
  → 排除四周 14 px 图像边界
  → 使用该图所属簇 high_threshold 二值化
  → 连通域分析
  → 保留面积 ≥ 64 px 的候选
```

每个候选记录：

```text
image_path
bbox_xywh
area_px
peak_anomaly
mean_anomaly
```

### 8.4 OK 污染候选

对 Stage 1 OK_Audit 图执行同一整图扫描，但不屏蔽 Polygon：

```text
OK Audit 图
  → 同簇、同位置多参考异常图
  → 本簇 high_threshold
  → 连通域候选
```

若出现稳定高异常峰值，表示：

- OK 图可能污染；
- OK-Bank 覆盖不足；
- 对齐仍存在场景差异；
- 正常背景具有多模态外观。

因此 OK 污染候选也必须人工确认，不能自动删除 OK 图。

---

## 9. 结果解释原则

### 9.1 疑似虚标不是确定错标

例如：

```text
Stage 2 weak annotation
```

只表示：

> 在同簇、多参考、同位置的 DINO Patch 比较下，Polygon 内视觉差异较弱。

仍可能是真实缺陷，尤其是：

- 目标尺寸很小；
- 对比度极弱；
- 纹理变化不明显；
- 缺陷依赖业务知识而非外观差异；
- DINO 感受野不适合该类微小缺陷。

### 9.2 疑似漏标不是自动补标

即使已做分簇和精确位置对齐，背景高异常仍可能来自：

- 反光、阴影、曝光变化；
- 正常背景多模态；
- 工件细微位移或姿态差异；
- 现有 Polygon 画小，导致异常外溢；
- 可接受的工艺纹理波动。

因此候选应按以下优先级人工复核：

1. 远离图像边缘；
2. 远离已有 Polygon 外扩安全区；
3. 连通域面积较大；
4. 峰值和均值均较高；
5. 在同类场景或多张图上重复出现；
6. 与中位数 OK 对照图差异直观明显。

---

## 10. 报告结构

Stage 2 人工审查报告目录：

```text
outputs/reports/stage2/
├── index.html
├── overview.svg
├── suspected_overlabel_gallery.html
├── coarse_or_misaligned_mask_gallery.html
├── missing_annotation_gallery.html
├── ok_bank_contamination_gallery.html
└── samples/
```

实例候选三联图：

```text
[左] 缺陷 Query Tile
[中] 多参考平均 Patch 异常热图 + Polygon 轮廓
[右] Stage 1 中位数距离 OK 参考 Tile
```

需要注意：

```text
右侧 OK 图只用于人工理解。
中间异常热图由该簇全部 OK-Bank 参考的平均 Patch 距离生成。
```

机器产物：

```text
outputs/artifacts/stage2/
├── stage2_summary.json
├── stage1_reference_metadata.json
├── instance_anomaly_metrics.json
├── missing_annotation_candidates.json
└── ok_bank_contamination_candidates.json
```

不输出 CSV，不持久化不必要的全尺寸浮点异常图，以避免产物膨胀。

---

## 11. 推理加速与显存策略

当前可用 GPU 显存约 `9 GB`，实现采用：

```text
- CUDA inference_mode
- CUDA FP16 autocast
- batch_size = 32
- 同簇/同位置多参考均值特征缓存
- Query/滑窗只执行一次 DINOv2 前向
- 只保留运行期异常图；报告仅写 PNG 面板和 JSON 指标
```

缓存层级：

```text
实例流：
  每个缺陷实例中心位置
  → 精确构建同簇全部 OK 参考均值 Patch

整图流：
  每个实际出现的滑窗归一化位置
  → 精确构建同簇全部 OK 参考均值 Patch
```

这种方式保留多参考平均距离的数学语义，同时避免：

```text
每一个 NG 滑窗
× 每一个同簇 OK 图
× 重复 DINOv2 前向
```

导致的数小时级运行时间。

---

## 12. 当前正式运行摘要

当前 Stage 2 正式运行：

```text
实例数量：439
OK 聚类数：2
Patch 特征：32 × 32 × 768
推理设备：CUDA
推理批量：32
```

候选数量：

| 类型 | 数量 |
|---|---:|
| 疑似虚标 / 视觉弱异常 | 43 |
| 疑似粗标 / 一框包多点 | 149 |
| 疑似边界错位 | 4 |
| 强视觉异常确认 | 39 |
| 疑似漏标候选 | 777 |
| OK 污染候选 | 10 |

这些数量用于排队审查，不等同于自动确认的错误标注数量。

---

## 13. 当前实现局限与后续改进

### 13.1 多参考均值会弱化少数正常模式

虽然多参考均值严格表达“平均距离”，但若同簇同位置存在多种正常外观，例如反光、阴影、纹理批次差异，均值可能不代表任一真实正常状态。

后续可扩展为：

```text
同簇、同位置
→ 保留多个正常子原型 / Medoid
→ 取最小或分位数距离
```

但这将改变“平均距离”的语义，应作为明确版本升级，而不是静默替换。

### 13.2 漏标候选仍需稳定性过滤

推荐在后续版本增加：

```text
- 多 OK 子参考的一致性评分；
- 候选相对周围环带的局部显著性；
- 跨图重复出现统计；
- 同类目标 ROI 白名单/背景掩膜；
- 机械模板配准或关键点对齐。
```

### 13.3 小缺陷的 Patch 感受野限制

一个 Patch 对应约 `14 × 14` 像素。极细小缺陷可能被正常背景稀释。

因此：

```text
Stage 2 weak annotation
+ Stage 0 Proto 高风险
```

不能直接视为虚标；它也可能是视觉可见但模型表征与分割尺度均不足的微小缺陷。

---

## 14. 运行方式

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
python scripts/run_stage2.py \
  --input-dir data \
  --stage1-artifact-dir outputs/artifacts/stage1 \
  --top-n 100
```

运行前要求：

```text
outputs/artifacts/stage1/
  - instance_queries.json
  - ok_bank_metadata.json
  - stage1_summary.json
```

必须来自新版 Stage 1，以保证场景分簇、簇内 Bank/Audit 切分及对齐语义一致。

# Stage 1：OK 视觉记忆库、机台/背景分簇与簇内动态阈值标定

## 1. 目标与边界

Stage 1 在 YOLO-Seg 训练前建立“正常外观参考系”，为每个**已标注缺陷实例**计算其局部上下文相对正常样本的偏离程度。

本阶段回答的问题是：

> 在同类机台、背景和产品结构位置中，该标注实例周围的视觉外观是否明显偏离正常状态？

Stage 1 的主用途是给人工审查排序，而不是自动删除标注：

- `normal`：接近 OK；疑似虚标、弱缺陷或类别灰区候选；
- `warning`：偏离正常但未明显越界；建议人工统一 OK/NG 标准；
- `high`：显著偏离正常；通常具有更强的异常视觉证据。

Stage 1 **不做**以下任务：

- 不扫描无标注区域，因此不能据此判定漏标；
- 不产生像素级异常热图，也不能证明异常精确位于 Polygon 内；
- 不替代 Stage 0 的 Proto 尺度可学习性检查。

像素级 Patch Token、Mask 内外异常对比和无标注区域扫雷属于后续 Stage 2。

---

## 2. 输入、输出与固定参数

### 输入

- LabelMe 格式的原图与同名 JSON；
- Stage 0 实例记录：Polygon、紧致 BBox、原图尺寸、Letterbox 参数、Proto 风险；
- `configs/stage1.json`。

### 当前固定模型与主要参数

| 项目 | 当前值 | 说明 |
|---|---:|---|
| 模型 | `dinov2_vitb14_reg` | 官方 `torch.hub` DINOv2 模型 |
| 图像特征 | L2 标准化 CLS Token | ViT-B/14 的 CLS 维度为 768 |
| 整图聚类输入 | Letterbox `448 × 448` | 保持原图宽高比，灰色填充 |
| 局部 Tile 尺寸 | `448 × 448` | Query、Bank、Audit 使用相同物理视野 |
| 相似度 | 余弦距离 | `d = 1 - cosine_similarity` |
| OK Audit 占比 | `20%` | 以重复组为切分单位 |
| 聚类算法 | KMeans | 仅由 OK 图确定簇中心 |
| K 选择 | Cosine Silhouette Score | 在可行候选 K 中取最大值 |
| 最小 OK 簇大小 | `8` | 小于该规模不能作为独立有效簇 |
| Audit 标定位置 | `3 × 3` 归一化位置网格 | 减少重叠滑窗重复计权 |

配置文件：[`configs/stage1.json`](../../configs/stage1.json)。

---

## 3. OK 图定义、去重与防泄露切分

### 3.1 OK 图定义

当前数据集规则：

```text
同名 LabelMe JSON 中不存在有效 Polygon 的图片，视为 OK 候选图。
```

“有效 Polygon”要求：

- `shape_type` 为 `polygon`；
- 顶点数不少于 3。

只要一张图含任意有效缺陷 Polygon，就不能进入 OK 候选集。

实现：[`src/features/ok_selection.py`](../../src/features/ok_selection.py)。

### 3.2 像素级重复图处理

OK 图使用文件字节的 SHA-256 建立重复组：

```text
同一 SHA-256 = 完全相同的像素级副本。
```

切分规则：

1. Bank/Audit 按重复组切分，任何重复组不能同时出现在两侧；
2. 每个重复组在 Bank 中只保留一个代表图；
3. Audit 同样每个重复组只保留一个代表图，避免同一视觉样本在分位数估计中被重复加权。

这避免了以下泄露：

```text
Audit 图 → 在 Bank 中存在完全相同副本 → 1-NN 距离接近 0 → 阈值虚低。
```

### 3.3 当前真实数据的 OK 样本情况

| 指标 | 数值 |
|---|---:|
| OK 候选图片 | 227 |
| 像素级唯一 OK 图像组 | 114 |
| 重复副本数 | 113 |
| 最终 OK-Bank 代表图 | 91 |
| 最终 OK-Audit 代表图 | 23 |

---

## 4. OK 整图自适应分簇

### 4.1 为什么需要先分簇

数据来自同一产品，但不同机台、背景、位置和视野条件不同。若将全部 OK 图混为一个全局库，会导致：

- 产品不同结构位置互相对照；
- 不同机台背景差异被误判为缺陷；
- 正常变化范围被混合，动态阈值失真；
- 已标注缺陷实例可能匹配到“最相似但位置错误”的 OK 局部区域。

因此，先由 OK 整图特征划分正常视觉域，再在每个视觉域内进行同位置对照。

### 4.2 整图特征提取

对每个去重后的 OK 代表图：

```text
原图
  → 等比例 Letterbox 到 448 × 448
  → DINOv2 ViT-B/14-Reg
  → x_norm_clstoken: [768]
  → L2 Normalize
```

整图 CLS 用于聚类与非 OK 图归簇，不直接作为缺陷异常得分。

### 4.3 自适应 K 选择

对于每一个候选 K：

```text
KMeans(random_state = fixed_seed, n_init = 20)
  → 检查每簇样本数 >= min_cluster_size
  → 计算 cosine silhouette score
```

选择 silhouette score 最大的有效 K：

```text
K* = argmax_K silhouette_cosine(K)
```

若样本数不足以支持至少 2 个有效簇，回退到：

```text
K = 1
```

并将原因写入 `fallback_reason`。

### 4.4 当前真实数据聚类结果

| K | Cosine Silhouette Score |
|---:|---:|
| 2 | **0.214965** |
| 3 | 0.204891 |
| 4 | 0.209957 |
| 5 | 0.186378 |
| 6 | 0.192108 |

最终：

```text
selected_k = 2
```

结果记录在：

- `outputs/artifacts/stage1/stage1_summary.json`
- `clustering.selected_k`
- `clustering.silhouette_scores`

实现：[`src/features/clustering.py`](../../src/features/clustering.py)。

---

## 5. 非 OK 图跟随 OK 簇

含缺陷标注的图不参与 KMeans，不修改 OK 簇中心。

对每张非 OK 图提取同样的 Letterbox-448 CLS 向量，并与 OK 簇中心计算余弦距离：

```text
d_c = 1 - cosine(non_ok_cls, ok_cluster_centroid_c)

assigned_cluster = argmin_c(d_c)
```

记录：

- `assigned_ok_cluster_id`：所属 OK 视觉簇；
- `distance_to_cluster_centroid`：到所属中心的距离；
- `second_cluster_distance`：次近簇距离；
- `cluster_assignment_ambiguous`：最近与次近距离差小于 `0.02` 时为真。

因此，非 OK 图只“跟随”正常视觉域，不会因异常外观把聚类边界拉偏。

---

## 6. 同簇、同位置 Tile 评估

### 6.1 缺陷 Query 的定位

对于每个 Stage 0 实例：

1. 获取 Polygon 紧致 BBox；
2. 计算 BBox 中心 `(cx, cy)`；
3. 以中心为锚点提取固定 `448 × 448` 上下文 Tile；
4. 靠近边缘时使用反射填充；
5. 计算中心归一化位置：

```text
u = cx / image_width
v = cy / image_height
```

Query Tile 与 Bank/Audit Tile 保持相同物理视野；避免可变 BBox 被强行放大到 `448 × 448` 而产生尺度不一致。

### 6.2 同簇同位置 OK Tile

若缺陷图属于 OK 簇 `c`，则只使用该簇中的 Bank 图。

对簇 `c` 的每一张 Bank 图：

```text
(u, v)
  → 映射到该 Bank 原图的对应像素位置
  → 提取以该位置为中心的固定 448 × 448 Tile
  → DINOv2 CLS 向量
```

该设计适用前提：产品不存在明显旋转、平移、缩放或未对齐；本项目数据满足该前提。

### 6.3 距离与主评分

对 Query CLS 向量 `q` 和簇内同位置的 `M` 个 OK CLS 向量 `{o_i}`：

```text
d_i = 1 - cosine(q, o_i)
```

主评分为距离均值：

```text
score_mean = (1 / M) * Σ d_i
```

同时保留：

| 字段 | 定义 | 用途 |
|---|---|---|
| `distance_mean` | `mean(d_i)` | Stage 1 主判定分数 |
| `distance_median` | `median(d_i)` | 稳健中心趋势 |
| `distance_std` | `std(d_i)` | 同位置 OK 外观稳定性 |
| `distance_min` | `min(d_i)` | 是否存在偶然相似 OK |
| `distance_max` | `max(d_i)` | 与极端 OK 外观的差异 |
| `bank_support` | `M` | 同簇可用 OK 对照图数 |

当前 `score_metric` 固定为：

```text
same_cluster_same_position_mean_cosine_distance
```

---

## 7. 簇级动态阈值标定

### 7.1 为什么阈值必须分簇

不同机台/背景/视野簇的正常变化幅度可能不同。若使用全局阈值：

- 外观稳定簇会被阈值放得过松；
- 外观变化较大簇会产生大量假高风险；
- 正常分布混合后难以解释。

因此每个 OK 簇独立执行 Audit 标定。

### 7.2 Audit 评分

对于每个簇的每张 Audit 代表图，在固定 `3 × 3` 归一化位置网格上取 `448 × 448` Tile。

每个 Audit Tile：

```text
仅与同簇 Bank 图的相同归一化位置 Tile 对照
  → 得到 {d_i}
  → 计算 audit_score = mean(d_i)
```

`3 × 3` 网格用于避免高度重叠滑窗把相近背景重复计入分位数；阈值仍严格对应“同簇、同位置、距离均值”的主评分定义。

### 7.3 阈值公式

对簇 `c` 的所有 Audit 分数：

```text
low_threshold_c = Percentile_95(audit_scores_c)

high_threshold_c = Percentile_99.5(audit_scores_c) × 1.2
```

判定：

```text
distance_mean <= low_threshold_c
    → normal：接近 OK / 疑似虚标候选

low_threshold_c < distance_mean <= high_threshold_c
    → warning：灰区 / 建议人工对齐

distance_mean > high_threshold_c
    → high：显著偏离 OK / 异常视觉证据强
```

### 7.4 当前真实数据阈值

| 簇 | Audit 样本数 | Low / P95 | High / P99.5 × 1.2 |
|---:|---:|---:|---:|
| Cluster 0 | 63 | 0.240537 | 0.320442 |
| Cluster 1 | 144 | 0.272735 | 0.462409 |

---

## 8. 中位数 OK Tile 可视化

报告不再展示“距离最小”的 OK Tile。

对于某一 Query 的所有 `d_i`：

```text
按 d_i 升序排序
  → 取排序中间位置对应的 OK Tile
  → 作为典型正常对照图
```

即：

```text
median_ok_tile = sorted_ok_tiles[len(sorted_ok_tiles) // 2]
```

它既不是偶然最相似的个体，也不是最不相似的极端个体，更适合代表同簇正常区域的典型外观。

每个报告卡片展示：

- 左图：缺陷实例中心的固定 `448 × 448` Query Tile；
- 右图：同簇、同归一化位置、距离中位数对应的典型 OK Tile；
- 指标：均值、中位数、标准差、簇级 low/high 阈值、簇 ID、Bank 支持数；
- Stage 0：Proto 风险等级和 `IoU_proto`；
- 典型 OK 图来源与归一化位置 `(u, v)`。

---

## 9. 结果解释与人工审查优先级

### Stage 1 分级解释

| Stage 1 | 统计含义 | 审查含义 |
|---|---|---|
| `normal` | `distance_mean <= low` | 更接近正常；优先检查虚标、弱缺陷和类别定义边界 |
| `warning` | `low < distance_mean <= high` | 灰区；建议由业务方定义 OK/NG 分界 |
| `high` | `distance_mean > high` | 明显偏离正常；更可能有有效视觉异常证据 |

### 与 Stage 0 联合解释

| Stage 0 | Stage 1 | 建议 |
|---|---|---|
| pass | normal | 首要“疑似虚标”复核候选 |
| pass | warning | 可学习但定义灰区，做标准对齐 |
| pass | high | 视觉明显、尺度可学的有效训练候选 |
| high/critical | normal | 不应直接删标；可能是微小弱缺陷且 Proto 难学 |
| high/critical | warning | 考虑切图、外扩 SOP 或豁免规则 |
| high/critical | high | 视觉异常明显但尺度难学，优先改训练/标注策略 |

### 当前真实运行结果

| 分级 | 实例数 |
|---|---:|
| `normal` | 253 |
| `warning` | 113 |
| `high` | 73 |

这些数字来自当前数据和当前随机种子，仅用于当前审查批次；数据、OK 构成或配置变化后必须重新运行 Stage 1。

---

## 10. 输出与可追溯性

### 人工审查产物

```text
outputs/reports/stage1/
├── index.html
├── nearest_neighbor_gallery.html
├── overview.svg
└── samples/
```

说明：`nearest_neighbor_gallery.html` 为历史文件名，但内部逻辑已不是全局最近邻；展示的是**同簇同位置距离中位数的典型 OK 对照**。

### 机器可读产物

```text
outputs/artifacts/stage1/
├── stage1_summary.json
├── instance_queries.json
├── ok_bank_metadata.json
└── ok_bank_vectors.npz
```

- `stage1_summary.json`：聚类候选分数、K、簇切分、簇级阈值、总计数；
- `instance_queries.json`：每个实例的簇归属、同位置距离统计、代表 OK Tile 和最终等级；
- `ok_bank_metadata.json`：Bank 图与 Audit 评分追溯信息；
- `ok_bank_vectors.npz`：压缩的向量矩阵。

项目不生成 CSV。

---

## 11. 代码入口与复现

| 文件 | 作用 |
|---|---|
| `scripts/run_stage1.py` | Stage 1 CLI 入口 |
| `src/features/dinov2_extractor.py` | DINOv2 CLS/Patch 特征提取与整图 Letterbox |
| `src/features/ok_selection.py` | OK 判定、重复组隔离切分 |
| `src/features/clustering.py` | 自适应 KMeans 与簇归属 |
| `src/features/ok_bank.py` | 固定 Tile 与同位置裁剪 |
| `src/features/stage1_runner.py` | Stage 1 编排、Audit 标定、实例评分 |
| `src/features/threshold_calibration.py` | 分位数动态阈值与等级判定 |
| `src/reports/stage1_report.py` | HTML/SVG/PNG 报告 |

运行：

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
python scripts/run_stage1.py --input-dir data --top-n 100
```

验证：

```bash
pytest -q
```

当前测试结果：

```text
15 passed
```

---

## 12. 当前限制与后续 Stage 2

1. OK 图定义基于“无 Polygon”，若其中含漏标缺陷，OK-Bank 仍可能被污染；
2. SHA-256 只能识别完全重复图，近重复图仍可能存在；
3. 使用 CLS 上下文分数，不等价于 Polygon 内部的像素级异常证据；
4. 产品虽已对齐，但若未来出现机台几何偏移，应在同位置映射前增加配准；
5. 小簇样本数量较少时，分位数阈值的统计稳定性会下降，应在报告中关注 `bank_support` 与 Audit 样本数。

Stage 2 将使用 `32 × 32 × 768` DINO Patch Token 特征，构建像素级异常图，并引入：

- Polygon Mask 加权池化；
- Mask 内外/环带差异；
- 粗标、错位、一框包多点分析；
- 已标注区域屏蔽后的无标注异常扫描；
- 更细粒度的虚标与漏标证据。

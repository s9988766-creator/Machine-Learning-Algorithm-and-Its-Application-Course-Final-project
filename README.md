# 乳房 X 光腫瘤分類 — 良性 vs 惡性

使用 CBIS-DDSM 乳房攝影資料集，比較**傳統機器學習（KNN + 手工特徵）**與**深度學習（CNN）**兩種路線在良性 / 惡性二分類任務上的表現。

---

## 資料集

- **來源**：[CBIS-DDSM (CLAHE Enhanced PNG 版本)](https://www.kaggle.com/datasets/dynemiesizumaki/cbis-ddsmpng)
- **大小**：約 2 GB
- **內容**：每個案例包含完整乳房 X 光 (`full_image.png`)、ROI 裁切影像 (`roi_cropped.png`)、ROI 遮罩 (`roi_mask.png`)
- **標籤組織**：依資料夾分類 `BENIGN/` 與 `MALIGNANT/`
- **案例數**：BENIGN 304、MALIGNANT 228

---

## 專案結構

```
breast-cancer-mammography-classification/
├── README.md
│
├── knn_handcrafted_features/         # 路線 1：傳統機器學習
│   ├── 01_feature_extraction.py      # 從 mask 抽取形狀特徵
│   └── 02_knn_classifier.py          # KNN 分類 + 特徵選擇 + 門檻調整
│
└── cnn_deep_learning/                # 路線 2：深度學習
    ├── 01_resnet50_baseline.ipynb               # ResNet50 + ImageNet 預訓練
    ├── 02_resnet18_bbox_crop.ipynb              # ResNet18 + bbox-crop 策略
    ├── 03_resnet50_radimagenet.ipynb            # ResNet50 + RadImageNet 預訓練
    ├── 04_resnet50_padding_preprocess.ipynb     # ResNet50 + padding 預處理
    └── 05_densenet121_padding_preprocess.ipynb  # DenseNet121 + padding 預處理
```

---

## 路線 1：KNN + 手工特徵

### 流程

1. **特徵抽取** (`01_feature_extraction.py`)：從 `roi_mask.png` 抽取以下形狀特徵
   - Area、Perimeter
   - Circularity（圓度）
   - Eccentricity（離心率）
   - Solidity（堅實度）
   - Aspect Ratio（長寬比）
   - Boundary Roughness（邊界粗糙度）
   - P/A Ratio（周長面積比）
   - 輸出：`features0512.csv`

2. **KNN 分類** (`02_knn_classifier.py`)：
   - 使用 `SequentialFeatureSelector` 做前向特徵選擇
   - 嘗試 k = 3, 5, 7, ..., 15 找最佳 k
   - 用 5-fold cross-validation 評估
   - 額外針對 **MALIGNANT recall** 做門檻調整（醫療上漏掉癌症代價較高）

### 結果

> 請填入你的 KNN 最終結果：
>
> | 指標 | 數值 |
> |---|---|
> | 最佳 k | ? |
> | Validation Accuracy | ? |
> | MALIGNANT Recall (調整門檻後) | ? |
> | 選用特徵 | (例如 perimeter, boundary_roughness, contrast, energy, entropy) |

---

## 路線 2：CNN 深度學習

### 共通設計

- **二分類任務**：BENIGN (0) vs MALIGNANT (1)
- **損失函數**：CrossEntropyLoss（加權重處理類別不平衡）或 BCEWithLogitsLoss
- **訓練策略**：Transfer learning（凍結 backbone 大部分，微調後段）
- **資料增強**：水平/垂直翻轉、旋轉、ColorJitter

### 五個實驗

| # | 檔案 | 模型 | 預處理 / 影像策略 | 切分方式 |
|---|---|---|---|---|
| 01 | `01_resnet50_baseline.ipynb` | ResNet50 (ImageNet) | `roi_cropped.png` 直接餵入 | **病人層級** (避免資料洩漏) |
| 02 | `02_resnet18_bbox_crop.ipynb` | ResNet18 (ImageNet) | 從 mask 找 bbox + 30% margin，裁切 `full_image.png` | **病人層級** |
| 03 | `03_resnet50_radimagenet.ipynb` | ResNet50 (RadImageNet) | 同上 bbox-crop | **病人層級** |
| 04 | `04_resnet50_padding_preprocess.ipynb` | ResNet50 (ImageNet) | `roi_cropped.png` + padding 補黑邊維持長寬比 | 影像層級 |
| 05 | `05_densenet121_padding_preprocess.ipynb` | DenseNet121 (ImageNet) | 同上 padding 預處理 | 影像層級 |

### 結果

| 實驗 | Test AUC | Test Accuracy | Sensitivity (MALIGNANT Recall) | Specificity |
|---|---|---|---|---|
| 01 ResNet50 baseline | 0.73 | 0.64 | 0.74 | 0.57 |
| **02 ResNet18 + bbox-crop** | **0.77** | 0.72 | 0.62 | 0.79 |
| 03 ResNet50 + RadImageNet | 0.67 | 0.63 | 0.69 | 0.60 |
| 04 ResNet50 + padding | — | 0.68 | 0.50 | — |
| 05 DenseNet121 + padding | — | 0.61 | 0.55 | — |

**最佳模型**：**02 ResNet18 + bbox-crop** (Test AUC = 0.77)

---

## 主要觀察與心得

1. **資料切分至關重要**
   - 病人層級切分 (patient-level split) 避免同病人影像同時出現在訓練與測試集
   - 影像層級切分會高估模型泛化能力

2. **影像策略影響大**
   - 直接餵 `roi_cropped` 效果尚可
   - 從 mask 抽取 bounding box + margin，從原圖裁切，能保留原始解析度 + 周圍組織 context
   - Padding 補黑邊維持長寬比的策略，在這個任務上效果不如 bbox-crop

3. **跨域預訓練不一定有幫助**
   - RadImageNet (放射科預訓練) 表現反而不如 ImageNet 預訓練
   - 推測原因：RadImageNet 以 CT/MRI/超音波為主，乳房 X 光佔比低
   - ImageNet 的通用低階特徵（邊緣、紋理）在小資料集上仍有競爭力

4. **類別不平衡 + 醫療場景偏好**
   - MALIGNANT 樣本少於 BENIGN，需要加 class weights
   - 醫學上漏掉癌症 (false negative) 的代價遠大於誤報，需要透過調整 threshold 拉高 sensitivity

---

## 環境需求

- Python 3.9+
- PyTorch、torchvision
- scikit-learn
- scikit-image（KNN 特徵抽取用）
- OpenCV
- pandas、numpy、matplotlib、PIL

```bash
pip install torch torchvision scikit-learn scikit-image opencv-python pandas matplotlib pillow
```

CNN 模型訓練建議使用 **Google Colab T4 GPU**（免費），單個 notebook 約 15-30 分鐘可訓練完成。

---

## 資料集下載 (CNN 用)

```bash
pip install kaggle
# 將 kaggle.json (從 https://www.kaggle.com/settings 取得) 放到 ~/.kaggle/
kaggle datasets download -d dynemiesizumaki/cbis-ddsmpng --unzip -p ./data
```

# Hệ Thống Phát Hiện Sự Kiện Âm Thanh Khẩn Cấp Sử Dụng Audio Spectrogram Transformer

## Mục Lục

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Kiến Trúc Hệ Thống](#2-kiến-trúc-hệ-thống)
3. [Cơ Sở Lý Thuyết](#3-cơ-sở-lý-thuyết)
4. [Thiết Lập Môi Trường](#4-thiết-lập-môi-trường)
5. [Chuẩn Bị Dữ Liệu](#5-chuẩn-bị-dữ-liệu)
6. [Thiết Kế Mô Hình](#6-thiết-kế-mô-hình)
7. [Huấn Luyện Mô Hình](#7-huấn-luyện-mô-hình)
8. [Đánh Giá và Kiểm Thử](#8-đánh-giá-và-kiểm-thử)
9. [Triển Khai và Sử Dụng](#9-triển-khai-và-sử-dụng)
10. [Kết Quả Thực Nghiệm](#10-kết-quả-thực-nghiệm)
11. [Tài Liệu Tham Khảo](#11-tài-liệu-tham-khảo)

---

## 1. Tổng Quan Dự Án

### 1.1. Giới Thiệu

Dự án này phát triển một hệ thống phát hiện sự kiện âm thanh khẩn cấp (Emergency Sound Event Detection) sử dụng kiến trúc **Audio Spectrogram Transformer (AST)**. Hệ thống có khả năng nhận diện và phân loại các âm thanh khẩn cấp trong môi trường đô thị như tiếng còi xe cứu thương, tiếng súng, tiếng kêu cứu, và các âm thanh nguy hiểm khác [1], [2], [3].

### 1.2. Mục Tiêu

- **Mục tiêu chính**: Xây dựng mô hình deep learning có độ chính xác cao cho bài toán phát hiện âm thanh khẩn cấp
- **Mục tiêu phụ**:
  - Tích hợp và tiền xử lý ba bộ dữ liệu chuẩn: UrbanSound8K, ESC-50.
  - Thiết kế kiến trúc AST tối ưu cho bài toán Sound Event Detection (SED)
  - Đạt hiệu suất real-time với độ trễ thấp cho ứng dụng thực tế
  - Xử lý vấn đề mất cân bằng dữ liệu và nhãn yếu (weakly labeled data)

### 1.3. Đóng Góp Chính

1. **Kiến trúc mô hình**: Triển khai AST với cơ chế patch embedding và multi-head self-attention cho dữ liệu âm thanh [4], [5]
2. **Xử lý dữ liệu**: Pipeline tiền xử lý và augmentation toàn diện cho ba bộ dữ liệu lớn
3. **Kỹ thuật huấn luyện**: Áp dụng Focal Loss, Mixup, SpecAugment, và mixed precision training [2], [6]
4. **Đánh giá toàn diện**: Metrics đa chiều bao gồm F1-score, mAP, latency, và confusion matrix

### 1.4. Ứng Dụng Thực Tiễn

- Hệ thống giám sát an ninh công cộng
- Phát hiện tình huống khẩn cấp trong thành phố thông minh
- Hỗ trợ người khiếm thị nhận biết nguy hiểm
- Tích hợp vào thiết bị IoT và edge computing [1], [7]

---

## 2. Kiến Trúc Hệ Thống

### 2.1. Cấu Trúc Thư Mục

```
audio-event-detection/
│
├── data/                          # Dữ liệu thô và đã xử lý
│   ├── raw/                       # Dữ liệu gốc
│   │   ├── UrbanSound8K/
│   │   ├── ESC-50/
│   │   └── FSD50K/
│   ├── processed/                 # Dữ liệu sau tiền xử lý
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── metadata/                  # File metadata và annotations
│
├── models/                        # Định nghĩa kiến trúc mô hình
│   ├── __init__.py
│   ├── ast_model.py              # Audio Spectrogram Transformer
│   ├── losses.py                 # Custom loss functions
│   └── pretrained/               # Pre-trained weights
│
├── utils/                         # Các module tiện ích
│   ├── __init__.py
│   ├── preprocess.py             # Tiền xử lý âm thanh
│   ├── augmentation.py           # Data augmentation
│   ├── dataset.py                # PyTorch Dataset classes
│   └── metrics.py                # Evaluation metrics
│
├── scripts/                       # Scripts huấn luyện và đánh giá
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Evaluation script
│   └── inference.py              # Inference và demo
│
├── configs/                       # Configuration files
│   └── config.yaml               # Hyperparameters và settings
│
├── notebooks/                     # Jupyter notebooks phân tích
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
├── results/                       # Kết quả thực nghiệm
│   ├── checkpoints/              # Model checkpoints
│   ├── logs/                     # Training logs
│   └── figures/                  # Biểu đồ và visualizations
│
├── docs/                          # Tài liệu bổ sung
│   ├── architecture.md
│   ├── dataset_guide.md
│   └── training_guide.md
│
├── requirements.txt               # Python dependencies
├── config.yaml                    # Main configuration
└── README.md                      # File này
```

### 2.2. Luồng Xử Lý Dữ Liệu

```
Audio Input (WAV/MP3)
        ↓
[1. Preprocessing]
    - Resampling (16kHz/32kHz)
    - Normalization
    - Silence removal
        ↓
[2. Feature Extraction]
    - Mel-Spectrogram (128 mel bins)
    - Log-scale transformation
    - Time-Frequency representation
        ↓
[3. Data Augmentation]
    - Time stretching
    - Pitch shifting
    - Noise injection
    - SpecAugment (frequency/time masking)
    - Mixup
        ↓
[4. Patch Embedding]
    - Split spectrogram into patches (16×16)
    - Linear projection to embedding dimension
    - Add positional encoding
        ↓
[5. Transformer Encoder]
    - Multi-head self-attention (12 heads)
    - Feed-forward networks
    - Layer normalization
    - Residual connections
        ↓
[6. Classification Head]
    - Global average pooling
    - Fully connected layers
    - Softmax activation
        ↓
Output: Class probabilities + Timestamps
```

### 2.3. Kiến Trúc Tổng Thể

Hệ thống được thiết kế theo kiến trúc modular với các thành phần độc lập:

1. **Data Module**: Xử lý và chuẩn bị dữ liệu từ nhiều nguồn
2. **Model Module**: Định nghĩa kiến trúc AST và các biến thể
3. **Training Module**: Pipeline huấn luyện với early stopping và checkpointing
4. **Evaluation Module**: Đánh giá toàn diện với nhiều metrics
5. **Inference Module**: Triển khai real-time và batch prediction

---

## 3. Cơ Sở Lý Thuyết

### 3.1. Audio Spectrogram Transformer (AST)

Audio Spectrogram Transformer là kiến trúc transformer được thiết kế đặc biệt cho dữ liệu âm thanh, lấy cảm hứng từ Vision Transformer (ViT) [4], [5], [8].

#### 3.1.1. Nguyên Lý Hoạt Động

**Bước 1: Chuyển đổi âm thanh thành spectrogram**

Tín hiệu âm thanh thô được chuyển đổi thành biểu diễn thời gian-tần số (time-frequency representation) thông qua Short-Time Fourier Transform (STFT) và Mel-scale filtering:

```
X_mel = MelFilterBank(STFT(audio))
X_log = log(X_mel + ε)
```

Kết quả là một mel-spectrogram có kích thước `(T, F)` với `T` là số frame thời gian và `F` là số mel bins (thường là 128) [4].

**Bước 2: Patch Embedding**

Spectrogram được chia thành các patches không chồng lấp, tương tự như cách ViT xử lý ảnh:

```
Patch size: 16×16 pixels
Number of patches: N = (T/16) × (F/16)
```

Mỗi patch được flatten và chiếu tuyến tính lên không gian embedding có chiều `d_model` (thường là 768):

```
z_0 = [x_patch1·E; x_patch2·E; ...; x_patchN·E]
```

Trong đó `E` là ma trận embedding có kích thước `(16×16, d_model)` [5], [12].

**Bước 3: Positional Encoding**

Để mô hình học được thông tin vị trí của các patches, positional encoding được thêm vào:

```
z_0 = z_0 + E_pos
```

Có hai loại positional encoding chính:
- **Learnable positional encoding**: Học từ dữ liệu
- **Sinusoidal positional encoding**: Sử dụng hàm sin/cos [12]

**Bước 4: Transformer Encoder**

Chuỗi embeddings đi qua `L` layers của Transformer Encoder (thường `L=12`):

```
z'_l = MSA(LN(z_l)) + z_l                    # Multi-head Self-Attention
z_l+1 = MLP(LN(z'_l)) + z'_l                 # Feed-forward Network
```

Trong đó:
- `MSA`: Multi-head Self-Attention với 12 heads
- `LN`: Layer Normalization
- `MLP`: 2-layer feed-forward network với GELU activation [4], [5]

**Bước 5: Classification**

Output của transformer được aggregate (thường dùng mean pooling) và đưa qua classification head:

```
y = Softmax(W · MeanPool(z_L) + b)
```

#### 3.1.2. Ưu Điểm của AST

1. **Khả năng học long-range dependencies**: Self-attention cho phép mô hình học mối quan hệ giữa các phần xa nhau trong spectrogram [4], [5]
2. **Transfer learning hiệu quả**: Có thể pre-train trên AudioSet (2M samples) và fine-tune cho task cụ thể [6], [8]
3. **Không cần inductive bias**: Không giả định cấu trúc cục bộ như CNN, linh hoạt hơn [4]
4. **Hiệu suất cao**: Đạt state-of-the-art trên nhiều benchmark như AudioSet, ESC-50 [4], [8], [29]

### 3.2. Sound Event Detection (SED)

Sound Event Detection là bài toán xác định **loại** và **thời điểm** xuất hiện của các sự kiện âm thanh trong một đoạn audio [2], [21].

#### 3.2.1. Phân Loại Bài Toán SED

**Strong-label SED**: Có thông tin chính xác về onset và offset time của mỗi event
**Weak-label SED**: Chỉ có thông tin về sự xuất hiện của event trong clip, không có timestamp [2], [18]

Dự án này tập trung vào **weak-label SED** do hạn chế của dữ liệu annotation.

#### 3.2.2. Thách Thức

1. **Polyphonic sounds**: Nhiều âm thanh xảy ra đồng thời
2. **Class imbalance**: Một số lớp có rất ít samples
3. **Background noise**: Nhiễu môi trường ảnh hưởng đến chất lượng
4. **Temporal localization**: Xác định chính xác thời điểm event [2], [3]

### 3.3. Hierarchical Token-Semantic Audio Transformer (HTS-AT)

HTS-AT là một biến thể cải tiến của AST với cấu trúc phân cấp (hierarchical) để giảm độ phức tạp tính toán [4], [27], [29].

#### 3.3.1. Kiến Trúc Phân Cấp

```
Stage 1: Patch size 4×4  → 384 dim
Stage 2: Patch size 2×2  → 768 dim  (merge tokens)
Stage 3: Patch size 1×1  → 768 dim  (merge tokens)
```

Mỗi stage giảm số lượng tokens xuống 1/4, giảm memory và tăng tốc độ [4], [29].

#### 3.3.2. Token-Semantic Module

Thay vì chỉ output class labels, HTS-AT tạo ra **class feature maps** cho phép localization trong thời gian:

```
Output shape: (num_classes, T', F')
```

Điều này cho phép xác định **khi nào** mỗi event xảy ra, không chỉ **có hay không** [29].

### 3.4. Kỹ Thuật Xử Lý Dữ Liệu

#### 3.4.1. SpecAugment

SpecAugment là kỹ thuật augmentation trực tiếp trên spectrogram, bao gồm [6]:

1. **Time Masking**: Che ngẫu nhiên `t` time frames liên tiếp
2. **Frequency Masking**: Che ngẫu nhiên `f` mel bins liên tiếp
3. **Time Warping**: Biến dạng trục thời gian

```python
# Ví dụ parameters
time_mask_param = 80      # Tối đa 80 frames
freq_mask_param = 27      # Tối đa 27 mel bins
num_masks = 2             # Số lần mask
```

#### 3.4.2. Mixup

Mixup tạo training samples ảo bằng cách kết hợp tuyến tính hai samples [2], [6]:

```
x_mixed = λ·x_i + (1-λ)·x_j
y_mixed = λ·y_i + (1-λ)·y_j
```

Trong đó `λ ~ Beta(α, α)` với `α=0.3` thường được sử dụng.

#### 3.4.3. Focal Loss

Focal Loss giải quyết vấn đề class imbalance bằng cách giảm trọng số của easy examples [2]:

```
FL(p_t) = -α_t(1 - p_t)^γ log(p_t)
```

Với `γ=2` và `α_t` là class weights, mô hình tập trung vào hard examples.

---

## 4. Thiết Lập Môi Trường

### 4.1. Yêu Cầu Hệ Thống

#### Phần Cứng Tối Thiểu
- **CPU**: Intel Core i5 hoặc AMD Ryzen 5 (4 cores)
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU với 6GB VRAM (GTX 1060 trở lên)
- **Storage**: 100 GB SSD khả dụng

#### Phần Cứng Khuyến Nghị
- **CPU**: Intel Core i7/i9 hoặc AMD Ryzen 7/9 (8+ cores)
- **RAM**: 32 GB trở lên
- **GPU**: NVIDIA RTX 3080/4080 (10GB+ VRAM) hoặc A100
- **Storage**: 500 GB NVMe SSD

#### Phần Mềm
- **OS**: Ubuntu 20.04/22.04 LTS hoặc Windows 10/11
- **Python**: 3.8, 3.9, hoặc 3.10
- **CUDA**: 11.7 hoặc 11.8 (cho GPU training)
- **cuDNN**: 8.x tương ứng với CUDA version

### 4.2. Cài Đặt Dependencies

#### Bước 1: Clone Repository

```bash
git clone https://github.com/your-username/audio-event-detection.git
cd audio-event-detection
```

#### Bước 2: Tạo Virtual Environment

**Sử dụng venv:**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

**Sử dụng conda:**
```bash
conda create -n audio-sed python=3.9
conda activate audio-sed
```

#### Bước 3: Cài Đặt PyTorch

**Với CUDA 11.8:**
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

**Với CPU only:**
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

#### Bước 4: Cài Đặt Các Thư Viện Khác

```bash
pip install -r requirements.txt
```

#### Bước 5: Kiểm Tra Cài Đặt

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import librosa; print(f'Librosa: {librosa.__version__}')"
```

### 4.3. Cấu Hình GPU

#### Kiểm Tra GPU

```bash
nvidia-smi
```

#### Thiết Lập CUDA Environment Variables

Thêm vào `~/.bashrc` hoặc `~/.zshrc`:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Sau đó:
```bash
source ~/.bashrc
```

---

## 5. Chuẩn Bị Dữ Liệu

### 5.1. Tổng Quan Các Bộ Dữ Liệu

Dự án sử dụng ba bộ dữ liệu chuẩn trong lĩnh vực Sound Event Detection:

| Dataset | Số Samples | Số Classes | Độ Dài Trung Bình | Sampling Rate | Đặc Điểm |
|---------|-----------|-----------|------------------|---------------|----------|
| **UrbanSound8K** | 8,732 | 10 | 4s | 44.1kHz | Urban sounds, 10-fold CV |
| **ESC-50** | 2,000 | 50 | 5s | 44.1kHz | Environmental sounds, 5-fold CV |
| **FSD50K** | 51,197 | 200 | Variable | 44.1kHz | Freesound, weakly labeled |

### 5.2. UrbanSound8K

#### 5.2.1. Giới Thiệu

UrbanSound8K chứa 8,732 đoạn âm thanh đô thị được gán nhãn, thuộc 10 lớp [1]:

1. **air_conditioner**: Tiếng máy điều hòa
2. **car_horn**: Tiếng còi xe hơi
3. **children_playing**: Tiếng trẻ em chơi
4. **dog_bark**: Tiếng chó sủa
5. **drilling**: Tiếng khoan
6. **engine_idling**: Tiếng động cơ nổ không tải
7. **gun_shot**: Tiếng súng (EMERGENCY)
8. **jackhammer**: Tiếng máy đục
9. **siren**: Tiếng còi xe cứu thương (EMERGENCY)
10. **street_music**: Tiếng nhạc đường phố

#### 5.2.2. Tải Xuống

**Cách 1: Tải thủ công**
```bash
# Truy cập: https://urbansounddataset.weebly.com/urbansound8k.html
# Tải file UrbanSound8K.tar.gz (6GB)
# Giải nén vào data/raw/
tar -xzf UrbanSound8K.tar.gz -C data/raw/
```

**Cách 2: Sử dụng script**
```bash
python scripts/download_datasets.py --dataset urbansound8k --output data/raw/
```

#### 5.2.3. Cấu Trúc Thư Mục

```
data/raw/UrbanSound8K/
├── audio/
│   ├── fold1/
│   │   ├── 100032-3-0-0.wav
│   │   ├── 100263-2-0-117.wav
│   │   └── ...
│   ├── fold2/
│   ├── ...
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

#### 5.2.4. Metadata Format

File `UrbanSound8K.csv` chứa các cột:

- `slice_file_name`: Tên file audio
- `fsID`: Freesound ID
- `start`: Thời điểm bắt đầu trong file gốc
- `end`: Thời điểm kết thúc
- `salience`: Độ nổi bật (1=foreground, 2=background)
- `fold`: Fold number (1-10) cho cross-validation
- `classID`: ID của class (0-9)
- `class`: Tên class

### 5.3. ESC-50

#### 5.3.1. Giới Thiệu

ESC-50 (Environmental Sound Classification) chứa 2,000 đoạn âm thanh môi trường thuộc 50 lớp, được tổ chức thành 5 nhóm chính:

1. **Animals**: 10 classes (dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow)
2. **Natural soundscapes & water sounds**: 10 classes (rain, sea waves, crackling fire, crickets, chirping birds, water drops, wind, pouring water, toilet flush, thunderstorm)
3. **Human, non-speech sounds**: 10 classes (crying baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing teeth, snoring, drinking/sipping)
4. **Interior/domestic sounds**: 10 classes (door knock, mouse click, keyboard typing, door/wood creaks, can opening, washing machine, vacuum cleaner, clock alarm, clock tick, glass breaking)
5. **Exterior/urban noises**: 10 classes (helicopter, chainsaw, siren, car horn, engine, train, church bells, airplane, fireworks, hand saw)

#### 5.3.2. Tải Xuống

```bash
# Clone repository
git clone https://github.com/karolpiczak/ESC-50.git data/raw/ESC-50

# Hoặc tải trực tiếp
wget https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip master.zip -d data/raw/
mv data/raw/ESC-50-master data/raw/ESC-50
```

#### 5.3.3. Cấu Trúc Thư Mục

```
data/raw/ESC-50/
├── audio/
│   ├── 1-100032-A-0.wav
│   ├── 1-100210-A-0.wav
│   └── ...
└── meta/
    └── esc50.csv
```

#### 5.3.4. Metadata Format

File `esc50.csv` chứa:

- `filename`: Tên file audio
- `fold`: Fold number (1-5)
- `target`: Class ID (0-49)
- `category`: Tên class
- `esc10`: True nếu thuộc ESC-10 subset
- `src_file`: File nguồn từ Freesound
- `take`: Take number

### 5.4. FSD50K

#### 5.4.1. Giới Thiệu

FSD50K (Freesound Dataset 50K) là bộ dữ liệu lớn với 51,197 clips thuộc 200 classes từ AudioSet ontology. Đây là bộ dữ liệu **weakly labeled** - chỉ có thông tin về sự xuất hiện của class trong clip, không có timestamp chính xác [2], [18].

#### 5.4.2. Tải Xuống

```bash
# Yêu cầu đăng ký tại: https://zenodo.org/record/4060432
# Tải các file sau:
# - FSD50K.dev_audio.zip (30GB)
# - FSD50K.eval_audio.zip (4GB)
# - FSD50K.ground_truth.zip (metadata)

# Giải nén
unzip FSD50K.dev_audio.zip -d data/raw/FSD50K/
unzip FSD50K.eval_audio.zip -d data/raw/FSD50K/
unzip FSD50K.ground_truth.zip -d data/raw/FSD50K/
```

#### 5.4.3. Cấu Trúc Thư Mục

```
data/raw/FSD50K/
├── FSD50K.dev_audio/
│   ├── 1.wav
│   ├── 2.wav
│   └── ...
├── FSD50K.eval_audio/
│   ├── 100001.wav
│   └── ...
└── FSD50K.ground_truth/
    ├── dev.csv
    ├── eval.csv
    ├── vocabulary.csv
    └── collection/
```

#### 5.4.4. Metadata Format

**dev.csv / eval.csv:**
- `fname`: Filename
- `labels`: Comma-separated list of labels (e.g., "Bark,Dog")
- `mids`: Comma-separated AudioSet MIDs
- `split`: train/val/test

**vocabulary.csv:**
- `mid`: AudioSet MID
- `display_name`: Human-readable label
- `description`: Description

### 5.5. Tiền Xử Lý Dữ Liệu

#### 5.5.1. Script Tiền Xử Lý

File `utils/preprocess.py` thực hiện các bước sau:

1. **Load audio**: Đọc file WAV/MP3
2. **Resampling**: Chuyển về sampling rate thống nhất (16kHz hoặc 32kHz)
3. **Normalization**: Chuẩn hóa amplitude về [-1, 1]
4. **Padding/Trimming**: Đưa về độ dài cố định
5. **Mel-spectrogram extraction**: Tạo mel-spectrogram
6. **Save processed data**: Lưu vào `data/processed/`

#### 5.5.2. Chạy Tiền Xử Lý

**Xử lý tất cả datasets:**
```bash
python utils/preprocess.py --config configs/config.yaml --datasets all
```

**Xử lý từng dataset riêng:**
```bash
# UrbanSound8K
python utils/preprocess.py --config configs/config.yaml --datasets urbansound8k

# ESC-50
python utils/preprocess.py --config configs/config.yaml --datasets esc50

# # FSD50K
# python utils/preprocess.py --config configs/config.yaml --datasets fsd50k
# ```

#### 5.5.3. Cấu Hình Tiền Xử Lý

Trong `configs/config.yaml`:

```yaml
preprocessing:
  sample_rate: 32000              # Target sampling rate
  duration: 10.0                  # Fixed duration in seconds
  n_fft: 1024                     # FFT window size
  hop_length: 320                 # Hop length for STFT
  n_mels: 128                     # Number of mel bins
  fmin: 50                        # Minimum frequency
  fmax: 14000                     # Maximum frequency
  normalize: true                 # Normalize audio
  trim_silence: true              # Remove leading/trailing silence
  top_db: 30                      # Threshold for silence removal
```

#### 5.5.4. Kiểm Tra Dữ Liệu Đã Xử Lý

```bash
python scripts/check_processed_data.py --data_dir data/processed/
```

Output mẫu:
```
Dataset Statistics:
-------------------
UrbanSound8K:
  - Total samples: 8,732
  - Train: 6,985 (80%)
  - Val: 873 (10%)
  - Test: 874 (10%)
  - Classes: 10
  - Spectrogram shape: (128, 1000)

ESC-50:
  - Total samples: 2,000
  - Train: 1,600 (80%)
  - Val: 200 (10%)
  - Test: 200 (10%)
  - Classes: 50
  - Spectrogram shape: (128, 1000)

FSD50K:
  - Total samples: 51,197
  - Train: 40,966 (80%)
  - Val: 5,115 (10%)
  - Test: 5,116 (10%)
  - Classes: 200
  - Spectrogram shape: (128, 1000)
```

### 5.6. Data Augmentation

#### 5.6.1. Các Kỹ Thuật Augmentation

File `utils/augmentation.py` triển khai các kỹ thuật sau:

**1. Time Stretching**
```python
# Thay đổi tốc độ phát mà không thay đổi pitch
rate = random.uniform(0.8, 1.2)  # 0.8x đến 1.2x
audio_stretched = librosa.effects.time_stretch(audio, rate=rate)
```

**2. Pitch Shifting**
```python
# Thay đổi pitch mà không thay đổi tốc độ
n_steps = random.uniform(-2, 2)  # ±2 semitones
audio_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
```

**3. Noise Injection**
```python
# Thêm white noise hoặc background noise
noise_factor = random.uniform(0.001, 0.005)
noise = np.random.randn(len(audio))
audio_noisy = audio + noise_factor * noise
```

**4. SpecAugment**
```python
# Time masking: Che ngẫu nhiên time frames
# Frequency masking: Che ngẫu nhiên mel bins
spec_augmented = spec_augment(
    spectrogram,
    time_mask_param=80,
    freq_mask_param=27,
    num_masks=2
)
```

**5. Mixup**
```python
# Trộn hai samples với tỷ lệ lambda
lambda_val = np.random.beta(0.3, 0.3)
mixed_audio = lambda_val * audio1 + (1 - lambda_val) * audio2
mixed_label = lambda_val * label1 + (1 - lambda_val) * label2
```

#### 5.6.2. Cấu Hình Augmentation

Trong `configs/config.yaml`:

```yaml
augmentation:
  train:
    time_stretch:
      enabled: true
      rate_range: [0.8, 1.2]
      prob: 0.5
    
    pitch_shift:
      enabled: true
      n_steps_range: [-2, 2]
      prob: 0.5
    
    noise_injection:
      enabled: true
      noise_factor_range: [0.001, 0.005]
      prob: 0.3
    
    spec_augment:
      enabled: true
      time_mask_param: 80
      freq_mask_param: 27
      num_masks: 2
      prob: 0.8
    
    mixup:
      enabled: true
      alpha: 0.3
      prob: 0.5
  
  val:
    # No augmentation for validation
    enabled: false
  
  test:
    # No augmentation for testing
    enabled: false
```

---

## 6. Thiết Kế Mô Hình

### 6.1. Kiến Trúc Audio Spectrogram Transformer

#### 6.1.1. Tổng Quan

File `models/ast_model.py` triển khai kiến trúc AST với các thành phần chính:

1. **Patch Embedding Layer**
2. **Positional Encoding**
3. **Transformer Encoder Blocks** (×12)
4. **Classification Head**

#### 6.1.2. Hyperparameters Chính

```yaml
model:
  name: "AudioSpectrogramTransformer"
  
  # Input configuration
  input_shape: [128, 1000]        # (n_mels, time_frames)
  patch_size: 16                  # Patch size for embedding
  
  # Transformer configuration
  embed_dim: 768                  # Embedding dimension
  num_heads: 12                   # Number of attention heads
  num_layers: 12                  # Number of transformer blocks
  mlp_ratio: 4.0                  # MLP hidden dim = embed_dim * mlp_ratio
  dropout: 0.1                    # Dropout rate
  attention_dropout: 0.1          # Attention dropout rate
  
  # Classification head
  num_classes: 10                 # Number of output classes (adjust per dataset)
  pooling: "mean"                 # Pooling method: "mean", "max", or "cls"
  
  # Positional encoding
  pos_encoding: "learnable"       # "learnable" or "sinusoidal"
```

#### 6.1.3. Chi Tiết Các Thành Phần

**1. Patch Embedding**

```python
class PatchEmbedding(nn.Module):
    def __init__(self, input_shape, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        
        # Convolutional projection
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, 1, H, W) - batch of spectrograms
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x
```

**2. Multi-Head Self-Attention**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x
```

**3. Transformer Block**

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x
```

**4. Complete AST Model**

```python
class AudioSpectrogramTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            config['input_shape'],
            config['patch_size'],
            config['embed_dim']
        )
        
        # Positional encoding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config['embed_dim']))
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config['embed_dim'],
                config['num_heads'],
                config['mlp_ratio'],
                config['dropout']
            )
            for _ in range(config['num_layers'])
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(config['embed_dim'])
        self.head = nn.Linear(config['embed_dim'], config['num_classes'])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (B, 1, H, W) - batch of spectrograms
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer encoder
        for block in self.blocks:
            x = block(x)
        
        # Normalization
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, embed_dim)
        
        # Classification
        x = self.head(x)  # (B, num_classes)
        return x
```

### 6.2. Loss Functions

#### 6.2.1. Focal Loss

File `models/losses.py` triển khai Focal Loss để xử lý class imbalance [2]:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights (tensor of shape [num_classes])
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # inputs: (B, num_classes) - logits
        # targets: (B,) - class indices
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

#### 6.2.2. Tính Class Weights

```python
def compute_class_weights(dataset, num_classes):
    """
    Compute class weights for imbalanced datasets
    """
    class_counts = np.zeros(num_classes)
    
    for _, label in dataset:
        class_counts[label] += 1
    
    # Inverse frequency weighting
    total_samples = len(dataset)
    class_weights = total_samples / (num_classes * class_counts)
    
    # Normalize
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return torch.FloatTensor(class_weights)
```

### 6.3. Model Variants

#### 6.3.1. AST-Small (Lightweight)

Cho edge devices và real-time applications:

```yaml
model_small:
  embed_dim: 384
  num_heads: 6
  num_layers: 6
  mlp_ratio: 4.0
  
  # Estimated parameters: ~22M
  # Inference time: ~15ms (GPU)
```

#### 6.3.2. AST-Base (Default)

Cân bằng giữa performance và efficiency:

```yaml
model_base:
  embed_dim: 768
  num_heads: 12
  num_layers: 12
  mlp_ratio: 4.0
  
  # Estimated parameters: ~86M
  # Inference time: ~30ms (GPU)
```

#### 6.3.3. AST-Large (High Performance)

Cho research và offline applications:

```yaml
model_large:
  embed_dim: 1024
  num_heads: 16
  num_layers: 24
  mlp_ratio: 4.0
  
  # Estimated parameters: ~300M
  # Inference time: ~80ms (GPU)
```

---

## 7. Huấn Luyện Mô Hình

### 7.1. Cấu Hình Huấn Luyện

#### 7.1.1. Training Hyperparameters

Trong `configs/config.yaml`:

```yaml
training:
  # Basic settings
  batch_size: 32                  # Batch size per GPU
  num_epochs: 100                 # Maximum number of epochs
  num_workers: 4                  # DataLoader workers
  
  # Optimizer
  optimizer:
    name: "AdamW"
    lr: 1.0e-4                    # Learning rate
    weight_decay: 0.05            # Weight decay for regularization
    betas: [0.9, 0.999]
    eps: 1.0e-8
  
  # Learning rate scheduler
  scheduler:
    name: "CosineAnnealingWarmRestarts"
    T_0: 10                       # Initial restart period
    T_mult: 2                     # Period multiplication factor
    eta_min: 1.0e-6               # Minimum learning rate
    warmup_epochs: 5              # Warmup epochs
    warmup_lr: 1.0e-6             # Initial warmup learning rate
  
  # Loss function
  loss:
    name: "FocalLoss"
    gamma: 2.0                    # Focusing parameter
    use_class_weights: true       # Use class weights for imbalance
  
  # Mixed precision training
  mixed_precision: true           # Use automatic mixed precision (AMP)
  
  # Gradient clipping
  grad_clip: 1.0                  # Max gradient norm
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 15                  # Stop if no improvement for 15 epochs
    min_delta: 0.001              # Minimum change to qualify as improvement
    monitor: "val_f1"             # Metric to monitor
    mode: "max"                   # "max" for metrics to maximize
  
  # Checkpointing
  checkpoint:
    save_best: true               # Save best model
    save_last: true               # Save last model
    save_top_k: 3                 # Save top-3 models
    monitor: "val_f1"
    mode: "max"
  
  # Logging
  logging:
    log_every_n_steps: 10
    use_tensorboard: true
    use_wandb: false              # Set to true if using Weights & Biases
    project_name: "audio-event-detection"
```

### 7.2. Training Pipeline

#### 7.2.1. Chạy Training

**Single GPU:**
```bash
python scripts/train.py --config configs/config.yaml --gpu 0
```

**Multi-GPU (DataParallel):**
```bash
python scripts/train.py --config configs/config.yaml --gpu 0,1,2,3
```

**Multi-GPU (DistributedDataParallel):**
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --config configs/config.yaml \
    --distributed
```

#### 7.2.2. Training Script Overview

File `scripts/train.py` thực hiện các bước sau:

```python
def train():
    # 1. Load configuration
    config = load_config('configs/config.yaml')
    
    # 2. Setup device and distributed training
    device = setup_device(config)
    
    # 3. Create datasets and dataloaders
    train_dataset = AudioDataset(config, split='train')
    val_dataset = AudioDataset(config, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # 4. Create model
    model = AudioSpectrogramTransformer(config['model'])
    model = model.to(device)
    
    # 5. Setup loss function
    if config['training']['loss']['use_class_weights']:
        class_weights = compute_class_weights(train_dataset, config['model']['num_classes'])
        class_weights = class_weights.to(device)
    else:
        class_weights = None
    
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=config['training']['loss']['gamma']
    )
    
    # 6. Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # 7. Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['training']['scheduler']['T_0'],
        T_mult=config['training']['scheduler']['T_mult'],
        eta_min=config['training']['scheduler']['eta_min']
    )
    
    # 8. Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
    
    # 9. Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        # Train one epoch
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, config
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        log_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics)
        
        # Save checkpoint
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_checkpoint(model, optimizer, epoch, val_metrics, 'best')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping']['patience']:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"Training completed. Best val F1: {best_val_f1:.4f}")
```

#### 7.2.3. Training Epoch

```python
def train_epoch(model, dataloader, criterion, optimizer, scaler, device, config):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (spectrograms, labels) in enumerate(pbar):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config['training']['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['grad_clip']
                )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if config['training']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['grad_clip']
                )
            
            optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, metrics
```

#### 7.2.4. Validation

```python
def validate(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for spectrograms, labels in tqdm(dataloader, desc="Validation"):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    return avg_loss, metrics
```

### 7.3. Transfer Learning

#### 7.3.1. Pre-trained Weights

Sử dụng pre-trained weights từ AudioSet để cải thiện performance [6], [8]:

```python
def load_pretrained_weights(model, pretrained_path):
    """
    Load pre-trained weights from AudioSet
    """
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Load state dict
    state_dict = checkpoint['model']
    
    # Remove classification head (different num_classes)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    
    print(f"Loaded pre-trained weights from {pretrained_path}")
    return model
```

#### 7.3.2. Fine-tuning Strategy

**Strategy 1: Freeze backbone, train head only**
```python
# Freeze all layers except classification head
for name, param in model.named_parameters():
    if 'head' not in name:
        param.requires_grad = False

# Train for 10 epochs
train(model, config, num_epochs=10)

# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Continue training with lower learning rate
config['training']['optimizer']['lr'] = 1e-5
train(model, config, num_epochs=50)
```

**Strategy 2: Gradual unfreezing**
```python
# Epoch 1-5: Train head only
# Epoch 6-10: Unfreeze last 4 transformer blocks
# Epoch 11-15: Unfreeze last 8 transformer blocks
# Epoch 16+: Unfreeze all layers
```

### 7.4. Monitoring Training

#### 7.4.1. TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir results/logs/ --port 6006

# Open browser: http://localhost:6006
```

Metrics được log:
- Training loss
- Validation loss
- Accuracy, Precision, Recall, F1-score
- Learning rate
- Gradient norms
- Confusion matrix (mỗi 5 epochs)

#### 7.4.2. Weights & Biases (Optional)

```python
import wandb

# Initialize
wandb.init(
    project="audio-event-detection",
    config=config,
    name=f"ast-{config['model']['name']}-{timestamp}"
)

# Log metrics
wandb.log({
    'train/loss': train_loss,
    'train/accuracy': train_acc,
    'val/loss': val_loss,
    'val/f1': val_f1,
    'lr': current_lr
})

# Log confusion matrix
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    probs=None,
    y_true=all_labels,
    preds=all_preds,
    class_names=class_names
)})
```

---

## 8. Đánh Giá và Kiểm Thử

### 8.1. Evaluation Metrics

File `utils/metrics.py` triển khai các metrics sau:

#### 8.1.1. Classification Metrics

```python
def compute_metrics(y_true, y_pred, y_prob=None, average='macro'):
    """
    Compute comprehensive classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for mAP)
        average: 'macro', 'micro', or 'weighted'
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, average_precision_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Mean Average Precision (mAP)
    if y_prob is not None:
        num_classes = y_prob.shape[1]
        y_true_onehot = np.eye(num_classes)[y_true]
        
        ap_per_class = []
        for i in range(num_classes):
            ap = average_precision_score(y_true_onehot[:, i], y_prob[:, i])
            ap_per_class.append(ap)
        
        metrics['mAP'] = np.mean(ap_per_class)
        metrics['AP_per_class'] = ap_per_class
    
    return metrics
```

#### 8.1.2. Latency Measurement

```python
def measure_latency(model, input_shape, device, num_runs=100, warmup=10):
    """
    Measure model inference latency
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, H, W)
        device: Device to run on
        num_runs: Number of inference runs
        warmup: Number of warmup runs
    
    Returns:
        Dictionary with latency statistics
    """
    import time
    
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'median': np.median(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99)
    }
```

### 8.2. Evaluation Script

#### 8.2.1. Chạy Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --config configs/config.yaml \
    --checkpoint results/checkpoints/best_model.pth \
    --split test \
    --output results/evaluation/

# Evaluate with specific dataset
python scripts/evaluate.py \
    --config configs/config.yaml \
    --checkpoint results/checkpoints/best_model.pth \
    --dataset urbansound8k \
    --split test
```

#### 8.2.2. Evaluation Output

Script tạo ra các file sau trong `results/evaluation/`:

1. **metrics.json**: Tất cả metrics dạng JSON
2. **confusion_matrix.png**: Confusion matrix visualization
3. **per_class_metrics.csv**: Metrics cho từng class
4. **classification_report.txt**: Detailed classification report
5. **roc_curves.png**: ROC curves cho từng class
6. **pr_curves.png**: Precision-Recall curves

### 8.3. Visualization

#### 8.3.1. Confusion Matrix

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot confusion matrix heatmap
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

#### 8.3.2. ROC Curves

```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curves(y_true, y_prob, class_names, save_path):
    """
    Plot ROC curves for all classes
    """
    num_classes = len(class_names)
    y_true_onehot = np.eye(num_classes)[y_true]
    
    plt.figure(figsize=(12, 8))
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr, tpr,
            label=f'{class_names[i]} (AUC = {roc_auc:.2f})',
            linewidth=2
        )
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves', fontsize=16)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### 8.4. Error Analysis

#### 8.4.1. Analyze Misclassifications

```python
def analyze_errors(model, dataloader, device, class_names, save_dir):
    """
    Analyze and visualize misclassified samples
    """
    model.eval()
    
    errors = []
    
    with torch.no_grad():
        for spectrograms, labels, file_paths in dataloader:
            spectrograms = spectrograms.to(device)
            outputs = model(spectrograms)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()
            
            # Find misclassifications
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    errors.append({
                        'file_path': file_paths[i],
                        'true_label': class_names[labels[i]],
                        'pred_label': class_names[preds[i]],
                        'confidence': F.softmax(outputs[i], dim=0).max().item()
                    })
    
    # Save error analysis
    error_df = pd.DataFrame(errors)
    error_df.to_csv(f'{save_dir}/misclassifications.csv', index=False)
    
    # Plot error distribution
    plt.figure(figsize=(12, 6))
    error_counts = error_df.groupby(['true_label', 'pred_label']).size()
    error_counts.plot(kind='bar')
    plt.title('Misclassification Distribution')
    plt.xlabel('(True Label, Predicted Label)')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_distribution.png', dpi=300)
    plt.close()
    
    return error_df
```

---

## 9. Triển Khai và Sử Dụng

### 9.1. Inference

#### 9.1.1. Batch Inference

```bash
# Inference on a directory of audio files
python scripts/inference.py \
    --config configs/config.yaml \
    --checkpoint results/checkpoints/best_model.pth \
    --input_dir /path/to/audio/files/ \
    --output_dir results/predictions/ \
    --batch_size 32
```

#### 9.1.2. Single File Inference

```python
from models.ast_model import AudioSpectrogramTransformer
from utils.preprocess import load_and_preprocess_audio
import torch

def predict_single_file(audio_path, model_path, config):
    """
    Predict class for a single audio file
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioSpectrogramTransformer(config['model'])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess audio
    spectrogram = load_and_preprocess_audio(audio_path, config)
    spectrogram = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    spectrogram = spectrogram.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(spectrogram)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    return {
        'class_id': pred_class,
        'class_name': config['class_names'][pred_class],
        'confidence': confidence,
        'all_probabilities': probs[0].cpu().numpy()
    }

# Usage
result = predict_single_file(
    'test_audio.wav',
    'results/checkpoints/best_model.pth',
    config
)
print(f"Predicted: {result['class_name']} (confidence: {result['confidence']:.2f})")
```

### 9.2. Real-time Detection

#### 9.2.1. Streaming Audio

```python
import pyaudio
import numpy as np
from collections import deque

class RealtimeAudioDetector:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # Audio stream settings
        self.sample_rate = config['preprocessing']['sample_rate']
        self.chunk_size = 1024
        self.buffer_duration = config['preprocessing']['duration']
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        
        # Circular buffer
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        # Convert bytes to numpy array
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to buffer
        self.audio_buffer.extend(audio_chunk)
        
        return (in_data, pyaudio.paContinue)
    
    def get_prediction(self):
        if len(self.audio_buffer) < self.buffer_size:
            return None
        
        # Get audio from buffer
        audio = np.array(self.audio_buffer)
        
        # Preprocess
        spectrogram = self.preprocess_audio(audio)
        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)
        spectrogram = spectrogram.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(spectrogram)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        return {
            'class_id': pred_class,
            'class_name': self.config['class_names'][pred_class],
            'confidence': confidence
        }
    
    def start(self):
        self.stream.start_stream()
        print("Real-time detection started...")
        
        try:
            while self.stream.is_active():
                prediction = self.get_prediction()
                if prediction and prediction['confidence'] > 0.7:
                    print(f"Detected: {prediction['class_name']} "
                          f"(confidence: {prediction['confidence']:.2f})")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()
    
    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

# Usage
detector = RealtimeAudioDetector(model, config, device)
detector.start()
```

### 9.3. Model Export

#### 9.3.1. Export to ONNX

```python
def export_to_onnx(model, config, output_path):
    """
    Export PyTorch model to ONNX format
    """
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(
        1, 1,
        config['preprocessing']['n_mels'],
        config['preprocessing']['duration'] * config['preprocessing']['sample_rate'] // config['preprocessing']['hop_length']
    )
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['spectrogram'],
        output_names=['logits'],
        dynamic_axes={
            'spectrogram': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")

# Usage
export_to_onnx(model, config, 'results/models/ast_model.onnx')
```

#### 9.3.2. ONNX Inference

```python
import onnxruntime as ort

def onnx_inference(audio_path, onnx_model_path, config):
    """
    Run inference using ONNX Runtime
    """
    # Load ONNX model
    session = ort.InferenceSession(onnx_model_path)
    
    # Preprocess audio
    spectrogram = load_and_preprocess_audio(audio_path, config)
    spectrogram = np.expand_dims(spectrogram, axis=(0, 1))  # (1, 1, H, W)
    
    # Run inference
    outputs = session.run(
        ['logits'],
        {'spectrogram': spectrogram.astype(np.float32)}
    )
    
    logits = outputs[0][0]
    probs = softmax(logits)
    pred_class = np.argmax(logits)
    
    return {
        'class_id': pred_class,
        'class_name': config['class_names'][pred_class],
        'confidence': probs[pred_class]
    }
```

#### 9.3.3. TorchScript Export

```python
def export_to_torchscript(model, config, output_path):
    """
    Export to TorchScript for C++ deployment
    """
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(
        1, 1,
        config['preprocessing']['n_mels'],
        config['preprocessing']['duration'] * config['preprocessing']['sample_rate'] // config['preprocessing']['hop_length']
    )
    
    # Trace model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save
    traced_model.save(output_path)
    
    print(f"TorchScript model saved to {output_path}")

# Usage
export_to_torchscript(model, config, 'results/models/ast_model.pt')
```

### 9.4. Web API

#### 9.4.1. Flask API

```python
from flask import Flask, request, jsonify
import base64
import io

app = Flask(__name__)

# Load model globally
model = load_model('results/checkpoints/best_model.pth', config)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for audio classification
    
    Request:
        - audio_file: Audio file (multipart/form-data)
        OR
        - audio_base64: Base64-encoded audio (JSON)
    
    Response:
        {
            "class_name": "siren",
            "confidence": 0.95,
            "all_probabilities": {...}
        }
    """
    try:
        # Get audio data
        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            audio_bytes = audio_file.read()
        elif 'audio_base64' in request.json:
            audio_base64 = request.json['audio_base64']
            audio_bytes = base64.b64decode(audio_base64)
        else:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Save to temporary file
        temp_path = '/tmp/temp_audio.wav'
        with open(temp_path, 'wb') as f:
            f.write(audio_bytes)
        
        # Predict
        result = predict_single_file(temp_path, model, config)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

#### 9.4.2. FastAPI (Recommended)

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Audio Event Detection API")

# Load model
model = load_model('results/checkpoints/best_model.pth', config)
model.eval()

@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    """
    Predict audio event class
    """
    try:
        # Save uploaded file
        temp_path = f'/tmp/{audio_file.filename}'
        with open(temp_path, 'wb') as f:
            content = await audio_file.read()
            f.write(content)
        
        # Predict
        result = predict_single_file(temp_path, model, config)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

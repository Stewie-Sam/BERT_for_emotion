import os
import numpy as np
import pandas as pd
import librosa
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')


# ===================== 1. 全局配置=====================
class Config:
    # 1. 数据集根目录（label.xlsx 所在文件夹）
    DATA_ROOT = "C:/Users/ASUS/Desktop/project/SIMS/SIMS"
    # 2. 标签（annotation列的3种值）
    EMOTION_MAP = {"Positive": 0, "Negative": 1, "Neutral": 2}
    # 3. Excel列名
    VIDEO_ID_COL = "video_id"  # 视频ID列
    TEXT_COL = "text"  # 文本内容列
    ANNOTATION_COL = "annotation"  # 情感标签列
    # 4. 特征/训练参数
    TEXT_DIM = 768  # BERT输出维度
    AUDIO_DIM = 128  # 音频MFCC特征维度
    IMAGE_DIM = 256  # 图像CNN特征维度
    NUM_CLASSES = len(EMOTION_MAP)
    BATCH_SIZE = 8
    EPOCHS = 30
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动用GPU/CPU


# 初始化配置
cfg = Config()

# ===================== 2. 单模态预处理模块（已适配你的数据）=====================
class TextProcessor:
    """文本预处理：直接从Excel的text列提取内容，用BERT生成特征"""

    def __init__(self):
        # 中文数据专用BERT模型
        MODEL_PATH = "C:/Users/ASUS/.cache/huggingface/hub/bert_base_chinese"
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        self.model = BertModel.from_pretrained(MODEL_PATH).to(cfg.DEVICE)
        self.model.eval()  # 仅特征提取，不训练

    def process(self, text_content):
        """直接传入Excel中的文本内容，无需读外部txt文件"""
        text = text_content.strip()
        # BERT编码
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=50,
            padding="max_length",
            truncation=True
        ).to(cfg.DEVICE)
        # 提取CLS句子级特征
        with torch.no_grad():
            outputs = self.model(**inputs)
            text_feat = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return text_feat


class AudioProcessor:
    """音频预处理：按 video_id_clip_id.wav 命名规则读取，提取MFCC特征"""

    def __init__(self):
        self.sample_rate = 16000  # 统一采样率
        self.n_mfcc = 64  # MFCC系数数量

    def process(self, audio_path):
        # 读取音频
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        # 预处理：去静音、归一化
        y, _ = librosa.effects.trim(y)  # 去除首尾静音
        y = librosa.util.normalize(y)  # 音量归一化
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=512, hop_length=256
        )
        # 特征聚合（均值+标准差→固定维度）
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        audio_feat = np.concatenate([mfcc_mean, mfcc_std])
        # 固定维度（不足补0，超过截断）
        if len(audio_feat) < cfg.AUDIO_DIM:
            audio_feat = np.pad(audio_feat, (0, cfg.AUDIO_DIM - len(audio_feat)))
        else:
            audio_feat = audio_feat[:cfg.AUDIO_DIM]
        return audio_feat


class ImageProcessor:
    """图像预处理：按 video_id_clip_id.jpg 命名规则读取，CNN提取特征"""

    def __init__(self):
        # 图像预处理流水线
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),  # 统一尺寸
            transforms.Grayscale(num_output_channels=1),  # 灰度化（减少计算）
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
        ])
        # 轻量级CNN特征提取器
        self.feat_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, cfg.IMAGE_DIM)
        ).to(cfg.DEVICE)
        self.feat_extractor.eval()

    def process(self, image_path):
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"图像读取失败：{image_path}（请检查文件路径和格式）")
        # 预处理+特征提取
        img_tensor = self.transform(img).unsqueeze(0).to(cfg.DEVICE)
        with torch.no_grad():
            img_feat = self.feat_extractor(img_tensor).squeeze().cpu().numpy()
        return img_feat


# ===================== 3. 批量预处理（自动关联Excel+音频+图像）=====================
def batch_preprocess():
    """批量处理所有数据，生成.npy特征文件（只需要运行1次）"""
    # 初始化处理器
    text_processor = TextProcessor()
    audio_processor = AudioProcessor()
    image_processor = ImageProcessor()

    # 读取你的 label.xlsx
    label_path = os.path.join(cfg.DATA_ROOT, "label.xlsx")
    df = pd.read_excel(label_path, engine="openpyxl")

    # 检查Excel必要列是否存在（防止列名错漏）
    required_cols = [cfg.VIDEO_ID_COL, cfg.TEXT_COL, cfg.ANNOTATION_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ label.xlsx 缺少必要列：{col}，请检查列名是否正确！")

    # 自动生成音频/图像路径（按命名规则：video_0001_1.wav/jpg）
    df["audio_path"] = df.apply(
        lambda row: os.path.join(cfg.DATA_ROOT, "audio", f"{row[cfg.VIDEO_ID_COL]}.wav"),
        axis=1
    )
    df["image_path"] = df.apply(
        lambda row: os.path.join(cfg.DATA_ROOT, "image", f"{row[cfg.VIDEO_ID_COL]}.jpg"),
        axis=1
    )

    # 存储特征和标签
    text_feats = []
    audio_feats = []
    image_feats = []
    labels = []

    # 逐个处理数据
    total = len(df)
    for idx, row in df.iterrows():
        try:
            # 生成唯一样本ID（video_id + clip_id）
            sample_id = row[cfg.VIDEO_ID_COL]
            # 从Excel直接获取文本内容
            text_content = row[cfg.TEXT_COL]
            audio_path = row["audio_path"]
            image_path = row["image_path"]
            emotion = row[cfg.ANNOTATION_COL]



            # 检查音频/图像文件是否存在
            for path, name in zip([audio_path, image_path], ["音频", "图像"]):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"❌ {name}文件不存在：{path}\n请按规则命名并放置文件！")

            # 处理各模态数据
            text_feat = text_processor.process(text_content)  # 直接用Excel文本
            audio_feat = audio_processor.process(audio_path)
            image_feat = image_processor.process(image_path)
            label = cfg.EMOTION_MAP[emotion]  # 标签转数字

            if idx < 10:
                print(f"\n第{idx + 1}个样本（sample_id={sample_id}）:")
                print("文本内容:", text_content)
                print("text_feat均值:  {:.6f}".format(np.mean(text_feat)))
                print("image_feat均值: {:.6f}".format(np.mean(image_feat)))
                print("audio_feat均值: {:.6f}".format(np.mean(audio_feat)))
                print("-" * 40)

            # 保存特征
            text_feats.append(text_feat)
            audio_feats.append(audio_feat)
            image_feats.append(image_feat)
            labels.append(label)

            # 打印进度
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                print(f"✅ 预处理进度：{idx + 1}/{total}（成功样本ID：{sample_id}）")

        except Exception as e:
            print(f"❌ 处理样本 {sample_id} 失败：{str(e)}，跳过该样本")
            continue

    # 保存预处理后的特征文件
    save_dir = os.path.join(cfg.DATA_ROOT, "processed")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "text_feats.npy"), np.array(text_feats))
    np.save(os.path.join(save_dir, "audio_feats.npy"), np.array(audio_feats))
    np.save(os.path.join(save_dir, "image_feats.npy"), np.array(image_feats))
    np.save(os.path.join(save_dir, "labels.npy"), np.array(labels))

    # 打印统计信息
    print(f"\n🎉 预处理完成！特征文件保存至：{save_dir}")
    print(f"📊 统计：总样本数{total} | 成功处理{len(labels)} | 失败{total - len(labels)}")


# ===================== 4. 多模态数据集 & 模型（无需修改）=====================
class MultimodalDataset(Dataset):
    """加载预处理后的特征，适配多模态输入"""

    def __init__(self, text_feats, audio_feats, image_feats, labels):
        self.text_feats = torch.tensor(text_feats, dtype=torch.float32)
        self.audio_feats = torch.tensor(audio_feats, dtype=torch.float32)
        self.image_feats = torch.tensor(image_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.text_feats[idx],
            self.audio_feats[idx],
            self.image_feats[idx],
            self.labels[idx]
        )


class MultimodalModel(nn.Module):
    """多模态融合模型（特征拼接+全连接，新手友好）"""

    def __init__(self):
        super().__init__()
        # 单模态特征投影（统一到256维）
        self.text_proj = nn.Linear(cfg.TEXT_DIM, 256)
        self.audio_proj = nn.Linear(cfg.AUDIO_DIM, 256)
        self.image_proj = nn.Linear(cfg.IMAGE_DIM, 256)
        # 融合分类层
        self.fusion_head = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止过拟合
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, cfg.NUM_CLASSES)
        )

    def forward(self, text_x, audio_x, image_x):
        # 特征投影
        text_x = self.text_proj(text_x)
        audio_x = self.audio_proj(audio_x)
        image_x = self.image_proj(image_x)
        # 多模态拼接（核心融合方式）
        fused_x = torch.cat([text_x, audio_x, image_x], dim=1)
        # 分类输出
        logits = self.fusion_head(fused_x)
        return logits


# ===================== 5. 模型训练 & 评估（无需修改）=====================
def train_model():
    """训练多模态情感模型（需先运行batch_preprocess）"""
    # 加载预处理后的特征
    feat_dir = os.path.join(cfg.DATA_ROOT, "processed")
    try:
        text_feats = np.load(os.path.join(feat_dir, "text_feats.npy"))
        audio_feats = np.load(os.path.join(feat_dir, "audio_feats.npy"))
        image_feats = np.load(os.path.join(feat_dir, "image_feats.npy"))
        labels = np.load(os.path.join(feat_dir, "labels.npy"))
    except FileNotFoundError:
        raise FileNotFoundError("❌ 未找到预处理特征文件！请先运行 batch_preprocess()")

    # 检查有效样本数
    if len(labels) == 0:
        raise ValueError("❌ 没有有效预处理特征，请检查数据和文件路径")

    # 划分训练集/测试集（8:2，分层抽样）
    (text_train, text_test,
     audio_train, audio_test,
     image_train, image_test,
     y_train, y_test) = train_test_split(
        text_feats, audio_feats, image_feats, labels,
        test_size=0.2, random_state=42, stratify=labels
    )

    # 构建数据集和DataLoader
    train_dataset = MultimodalDataset(text_train, audio_train, image_train, y_train)
    test_dataset = MultimodalDataset(text_test, audio_test, image_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # 初始化模型、优化器、损失函数
    model = MultimodalModel().to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = nn.CrossEntropyLoss()  # 多分类损失

    # 训练循环
    best_test_acc = 0.0
    for epoch in range(cfg.EPOCHS):
        # 训练模式
        model.train()
        train_loss = 0.0
        train_preds = []
        train_true = []

        for batch in train_loader:
            text_x, audio_x, image_x, y = [x.to(cfg.DEVICE) for x in batch]
            # 前向传播
            logits = model(text_x, audio_x, image_x)
            loss = criterion(logits, y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录损失和预测
            train_loss += loss.item() * text_x.size(0)
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_true.extend(y.cpu().numpy())

        # 计算训练集指标
        train_loss = train_loss / len(train_dataset)
        train_acc = accuracy_score(train_true, train_preds)

        # 测试集评估
        model.eval()
        test_preds = []
        test_true = []
        with torch.no_grad():
            for batch in test_loader:
                text_x, audio_x, image_x, y = [x.to(cfg.DEVICE) for x in batch]
                logits = model(text_x, audio_x, image_x)
                test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                test_true.extend(y.cpu().numpy())
        test_acc = accuracy_score(test_true, test_preds)

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(feat_dir, "best_multimodal_model.pth"))
            print(f"📌 新最佳模型保存（测试准确率：{best_test_acc:.4f}）")

        # 打印训练日志
        print(f"Epoch [{epoch + 1}/{cfg.EPOCHS}] | "
              f"训练损失：{train_loss:.4f} | "
              f"训练准确率：{train_acc:.4f} | "
              f"测试准确率：{test_acc:.4f} | "
              f"最佳准确率：{best_test_acc:.4f}")

    # 训练完成：输出详细分类报告
    print("\n" + "=" * 60)
    print(f"🎯 训练完成！最佳测试准确率：{best_test_acc:.4f}")
    # 加载最佳模型重新评估
    best_model = MultimodalModel().to(cfg.DEVICE)
    best_model.load_state_dict(torch.load(os.path.join(feat_dir, "best_multimodal_model.pth")))
    best_model.eval()
    with torch.no_grad():
        final_preds = []
        final_true = []
        for batch in test_loader:
            text_x, audio_x, image_x, y = [x.to(cfg.DEVICE) for x in batch]
            logits = best_model(text_x, audio_x, image_x)
            final_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            final_true.extend(y.cpu().numpy())
    # 标签反向映射（数字→文字）
    inv_emotion_map = {v: k for k, v in cfg.EMOTION_MAP.items()}
    final_true_labels = [inv_emotion_map[x] for x in final_true]
    final_pred_labels = [inv_emotion_map[x] for x in final_preds]
    print("\n📋 详细分类报告：")
    print(classification_report(final_true_labels, final_pred_labels))


# ===================== 6. 主函数（控制流程，无需修改）=====================
if __name__ == "__main__":
    # 第一步：运行批量预处理
    batch_preprocess()
    # 第二步：运行模型训练
    train_model()
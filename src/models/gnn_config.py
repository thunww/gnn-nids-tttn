from pathlib import Path

# Tang 64->128 (2026-07-19): them dung luong cho bai toan 15 lop (CSE-CIC) + dac trung node
# vua tang tu 4 len 43 chieu -- xem docs/decisions.md.
HIDDEN_DIM = 128
NUM_LAYERS = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4  # L2 regularization, chong overfitting (dac biet cho GAT)
BATCH_SIZE = 32
GAT_HEADS = 4

# Dropout rieng cho tung kien truc -- GAT can dropout cao hon de chong overfitting
# tren he so attention (tham khao ReGrAt, arXiv:2211.14770)
DROPOUT_GCN = 0.4
DROPOUT_GAT = 0.5

MAX_EPOCHS = 80
# patience tang tu 5 -> 15 (2026-07-19): voi bo du lieu it batch/epoch (vd UNSW-NB15 chi ~10
# batch/epoch), val_f1_macro rat nhieu tung epoch -- patience ngan de kich hoat nham do nhieu
# ngau nhien, khoa cung mo hinh vao trang thai doan bua 1 lop qua som. Xem docs/decisions.md.
EARLY_STOPPING_PATIENCE = 15

# Class-Balanced Loss (Cui et al., CVPR 2019) -- he so beta cho cong thuc "so mau hieu qua"
CB_BETA = 0.999

# Tu giam learning rate khi val_f1_macro chung lai (ReduceLROnPlateau) -- GNN von noi tieng
# huan luyen khong on dinh, scheduler giup hoi tu muot hon. patience tang 3 -> 8 cung ly do tren.
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 8

# Tang 4->43 (2026-07-19): 4 dac trung cau truc (bac vao/ra, pagerank, clustering) + 39 dac
# trung tong hop tu canh ke (trung binh cac canh vao/ra node do) -- xem graph/node_features.py,
# docs/decisions.md ("lam giau dac trung node"). Node2Vec/line-graph (cach "chuan" hon trong
# tai lieu N2V-EGS-PCA) khong kha thi trong pipeline theo-cua-so hien tai, day la thay the thuc te.
NODE_FEATURE_DIM = 43
EDGE_FEATURE_DIM = 39  # so dac trung luong mang da chuan hoa (giong feature_cols o ETL/Graph Builder)

DEFAULT_PROCESSED_DIR = Path("data/processed")

# Transfer learning (2026-07-19): UNSW-NB15-v2 qua it do thi (668 train) de tu hoc phan biet
# cac lop chong lan dac trung (xem docs/decisions.md) -- nap truoc trong so da hoc tu
# CSE-CIC-IDS2018 (13.224 do thi, da hoc tot cach nhan dien cau truc tan cong noi chung) lam
# diem khoi dau, thay vi khoi tao ngau nhien. Day la THU NGHIEM BO SUNG cho RQ1 (within-dataset),
# KHAC voi Thi nghiem 2/RQ2 (train 1 bo, test thang sang bo kia, khong tinh chinh).
PRETRAINED_SOURCE = {"nf-unsw-nb15-v2": "nf-cse-cic-ids2018-v2"}
FINE_TUNE_LEARNING_RATE = LEARNING_RATE / 10

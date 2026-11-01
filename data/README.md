# Benchmark Datasets & Preprocessing Protocol

Thư mục này chứa toàn bộ các tập dữ liệu phục vụ nghiên cứu mô hình ngôn ngữ nhỏ cho bài toán Text-to-SQL Semantic Parsing.

## 1. Nguồn dữ liệu (Data Sources)
* **Yale Spider Benchmark** (Yu et al., EMNLP 2018): Bộ dữ liệu phức tạp đa miền với 1.034 câu hỏi Dev Set và 166 CSDL SQLite.
* **Spider-Realistic** (Deng et al., 2021): Tập dữ liệu 508 câu hỏi thực tế loại bỏ sự trùng lặp từ khóa cột nhằm đánh giá năng lực Schema Linking thực sự.

## 2. Nguyên tắc Cách ly CSDL (Strict Out-of-Domain Generalization)
Để đảm bảo tính khách quan học thuật, tập huấn luyện `cot_train_800.json` (800 mẫu) được chọn lọc và cách ly 100% đối với toàn bộ các CSDL có mặt trong tập kiểm định `spider_dev.json` và `spider_realistic.json`.

## 3. Lệnh tái lập dữ liệu (Data Pipeline Execution)
Chạy 2 lệnh sau từ thư mục gốc của repository:
```bash
# Bước 1: Tải và giải nén CSDL SQLite cùng tập Dev
python data/download_and_prepare.py

# Bước 2: Sinh tập dữ liệu huấn luyện CoT chuẩn hóa
python data/preprocess_cot_train.py

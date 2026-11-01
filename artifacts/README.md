# Experimental Artifacts & Evaluation Logs

Thư mục này chứa toàn bộ các chứng cứ số liệu thực nghiệm nguyên bản phục vụ bài báo.

## 1. Danh mục Tệp Checkpoint Thực nghiệm (`results_json/`)
Toàn bộ kết quả thực thi từng câu hỏi trên SQLite engine được lưu trữ dưới dạng JSON có cấu trúc:
* `stage1_eval_checkpoint.json`: Chi tiết 1.034 câu hỏi Dev Set đánh giá 4 phương pháp ban đầu (M1 - M4).
* `stage2_self_repair_results.json`: Chi tiết thực nghiệm vòng lặp tự sửa lỗi (Self-Repair) đối chứng.
* `spider_original_cot_results.json`: Log thực nghiệm của mô hình M4-CoT tích hợp Repair-CoT trên Spider gốc (Đạt 64.70% EX).
* `spider_m2_original_cot_results.json`: Log thực nghiệm M2-CoT trên Spider gốc (Đạt 32.98% EX).
* `spider_realistic_cot_results.json`: Log thực nghiệm kiểm chứng khả năng tổng quát hóa trên Spider-Realistic (508 câu).

## 2. Trọng số Mô hình LoRA (`checkpoints/lora_adapter/`)
Chứa cấu hình và trọng số adapter tinh chỉnh tham số của mô hình `Qwen2.5-Coder-1.5B-Instruct` sau khi huấn luyện trên các chuỗi suy luận CoT.

## 3. Bảng số liệu tổng hợp (`results_csv/`)
Các bảng CSV được xuất tự động từ log JSON để đối soát trực tiếp với các bảng trong bài báo khoa học.

# BÁO CÁO THỰC HÀNH LAB 16: CLOUD AI ENVIRONMENT SETUP (GCP)

**Học viên:** Phạm Tấn Gia Quốc  
**Mã học viên:** 2A202601606  
**Nền tảng:** Google Cloud Platform (GCP)  
**Project ID:** `track2-day16-2a202601606`  
**Region / Zone:** `us-central1` / `us-central1-a`  
**Cấu hình VM:** `e2-medium` (2 vCPU, 4GB RAM, Debian 12)  
**Ngày thực hiện:** 14/08/2026  

---

## 1. Kết quả Huấn luyện & Đánh giá Mô hình (LightGBM)

| Chỉ số (Metric) | Kết quả thực tế | Ý nghĩa & Ghi chú |
| :--- | :---: | :--- |
| **Tập dữ liệu (Dataset)** | Credit Card Fraud Detection | 284,807 dòng, 31 đặc trưng (Kaggle chuẩn) |
| **Thời gian nạp dữ liệu (Load Time)** | **2.0724 s** | Đọc tệp `creditcard.csv` dung lượng 143 MB |
| **Thời gian huấn luyện (Training Time)** | **3.0892 s** | Huấn luyện trên 227,845 mẫu (80% dataset) |
| **AUC-ROC** | **0.857680** | Năng lực phân biệt giao dịch bình thường vs gian lận |
| **Độ chính xác (Accuracy)** | **99.8560%** | Chỉ số bị chi phối bởi lớp đa số (99.83% bình thường) |
| **Recall (Độ nhạy)** | **70.4082%** | Bắt được hơn 70% các vụ gian lận thực tế |
| **Precision (Độ chuẩn xác)** | **56.5574%** | Tỷ lệ dự đoán đúng trong các ca cảnh báo gian lận |
| **F1-Score** | **0.627273** | Trung bình điều hòa giữa Precision và Recall |
| **Độ trễ dự đoán 1 dòng (Latency)** | **0.9228 ms** | Trung bình qua 100 lần chạy (sau 10 lần warm-up) |
| **Thông lượng suy luận (Throughput)** | **320,575.09 rows/s** | Xử lý theo lô (batch 1,000 dòng) trên CPU |

---

## 2. Toàn văn Kết quả Chuẩn (`benchmark_result.json`)

```json
{
  "cloud": "gcp",
  "instance_type": "e2-medium",
  "dataset_rows": 284807,
  "load_time_seconds": 2.0724,
  "training_time_seconds": 3.0892,
  "auc_roc": 0.85768,
  "accuracy": 0.99856,
  "precision": 0.565574,
  "recall": 0.704082,
  "f1_score": 0.627273,
  "inference_latency_ms_one_row": 0.9228,
  "inference_throughput_rows_per_second": 320575.09
}
```

---

## 3. Báo cáo Phân tích Chuyên môn 6 Điểm Cốt lõi

1. **Thời gian triển khai & Bootstrap:** Quá trình tự động hóa bằng Terraform để khởi tạo toàn bộ hạ tầng mạng (VPC, Private Subnet, Cloud Router, Cloud NAT, IAP Firewall, Load Balancer) và máy ảo Compute Node `e2-medium` hoàn tất sau **~2 phút 30 giây**. Script startup tự động bootstrap cài đặt môi trường Machine Learning mất thêm **~1 phút**.
2. **Huấn luyện & Hiệu năng mô hình:** Trên CPU máy ảo `e2-medium` (2 vCPU, 4GB RAM), mô hình LightGBM chỉ mất **3.09 giây** để hoàn thành quá trình fit trên 227,845 mẫu huấn luyện và đạt diện tích dưới đường cong ROC (**AUC-ROC**) là **0.857680**.
3. **Phân tích "Bẫy dữ liệu mất cân bằng" (Imbalanced Data Trap):** Mặc dù Accuracy đạt mức rất cao là **99.86%**, chỉ số này gây ảo tưởng vì gian lận chỉ chiếm **0.173%** dataset. Đánh giá chất lượng thực tế dựa trên **Recall = 70.41%** (bắt được >70% ca gian lận thực tế, bỏ sót ~29.59%) và **Precision = 56.56%**, cho ra **F1-Score = 0.6273** phản ánh mức độ cân bằng tốt cho bài toán thực tế.
4. **Độ trễ và Thông lượng Inference:** Độ trễ dự đoán đơn dòng cực thấp (**0.92 ms/dòng**), hoàn toàn đáp ứng các hệ thống phát hiện gian lận thời gian thực (Real-time API). Khi gom theo lô (batch 1,000 dòng), thông lượng xử lý tăng vọt lên **320,575 dòng/giây** nhờ tối ưu hóa tính toán vector và SIMD đa luồng trên CPU.
5. **Nghẽn phần cứng (Hardware Bottlenecks):** CPU và RAM hoàn toàn không bị nghẽn: bộ nhớ RAM chỉ chiếm dụng **~470 MB / 3.8 GB** (còn trống 3.4 GB khả dụng) và CPU 2 vCPU xử lý mượt mà toàn bộ pipeline mà không gặp lỗi tràn bộ nhớ (OOM).
6. **Quản trị chi phí Cloud & Dọn dẹp:** Trong luồng CPU này, **Cloud NAT Gateway** và **Global Forwarding Rule IP của Load Balancer** là hai thành phần đóng góp chi phí duy trì cố định chính (~$0.052/giờ), cao hơn chi phí của VM `e2-medium` (~$0.033/giờ). Toàn bộ 16 tài nguyên đã được chạy `terraform destroy` thành công (`Destroy complete!`) và xác nhận không còn tài nguyên nào phát sinh chi phí trên GCP.

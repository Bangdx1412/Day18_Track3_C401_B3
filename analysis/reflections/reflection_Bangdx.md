# Individual Reflection — Lab 18

**Tên:** Đặng Xuân Bằng  
**Module phụ trách:** M2: Hybrid Search

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** Module 2 - Hybrid Search.
- **Các hàm/class chính đã viết:** 
    - `segment_vietnamese()`: Sử dụng `underthesea` để tách từ tiếng Việt.
    - `BM25Search`: Lập chỉ mục và tìm kiếm bằng thuật toán BM25 (Rank-BM25).
    - `DenseSearch`: Tích hợp Qdrant và SentenceTransformers (model BGE-M3) để tìm kiếm vector.
    - `reciprocal_rank_fusion()`: Kết hợp kết quả từ BM25 và Dense Search bằng thuật toán RRF.
    - `HybridSearch`: Lớp bao bọc (wrapper) để chạy luồng tìm kiếm hỗn hợp.
- **Số tests pass:** 5/5 (Tất cả các test case cho Module 2 đã vượt qua thành công).

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Reciprocal Rank Fusion (RRF) - một cách cực kỳ hiệu quả để kết hợp kết quả từ nhiều nguồn tìm kiếm khác nhau mà không cần chuẩn hóa điểm số (normalization). Hiểu rõ hơn về cách Qdrant lưu trữ và truy vấn vector.
- **Điều bất ngờ nhất:** Thư viện `underthesea` có sự khác biệt về kết quả tách từ giữa cụm từ ngắn và cụm từ dài (ví dụ: "nghỉ phép" vs "nghỉ phép năm"), điều này gây ảnh hưởng trực tiếp đến độ chính xác của BM25.
- **Kết nối với bài giảng (slide nào):** Kết nối trực tiếp với phần Retrieval (Slide về Hybrid Search, BM25 vs Dense Search).

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** 
    1. Cài đặt các thư viện nặng như `sentence-transformers` và `torch` trên môi trường Windows thường gặp lỗi xung đột phiên bản.
    2. Xử lý logic tách từ tiếng Việt để đảm bảo BM25 hoạt động hiệu quả.
- **Cách giải quyết:** 
    1. Sử dụng `pip install` cụ thể cho từng thư viện và kiểm tra `ModuleNotFoundError` để bổ sung kịp thời. Tạo tệp `requirements.txt` để quản lý.
    2. Debug chi tiết từng bước tokenization để hiểu tại sao BM25 score lại bằng 0.
- **Thời gian debug:** Khoảng 1 giờ cho các vấn đề về môi trường và tokenization.

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** Sẽ tìm cách xử lý tokenization nhất quán hơn, có thể là lowercase và loại bỏ dấu câu triệt để hơn trước khi đưa vào BM25, hoặc sử dụng thêm N-gram.
- **Module nào muốn thử tiếp:** Module 3 (Reranking) để xem kết quả Hybrid Search được cải thiện như thế nào sau khi dùng Cross-Encoder.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 4 |
| Teamwork | 5 |
| Problem solving | 4 |

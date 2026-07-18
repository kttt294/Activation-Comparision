# Báo cáo Thực nghiệm: Ảnh hưởng của Hàm Kích hoạt

Đây là mã nguồn thực nghiệm cho bài tập lớn môn **Nhập môn Trí tuệ Nhân tạo**. Toàn bộ mã nguồn trên Jupyter Notebook ban đầu đã được tái cấu trúc (refactor) thành một dự án Python chuyên nghiệp chuẩn Module nhằm tăng tính linh hoạt, khả năng tái sử dụng và dễ dàng kiểm soát lỗi.

## Cấu trúc Kiến trúc dự án
- `config.py`: File cấu hình tập trung. Chứa các tham số huấn luyện cốt lõi (Epochs, Batch size, LR) để thay đổi nhanh mà không làm hỏng logic.
- `data_loader.py`: Đóng gói logic tải dữ liệu MNIST và tự động xử lý chia tập Train/Validation.
- `models.py`: Khởi tạo khối mạng Multi-Layer Perceptron (MLP) động và định nghĩa các hàm kích hoạt.
- `engine.py`: Chứa "động cơ" (vòng lặp huấn luyện), đánh giá mô hình, thu thập số liệu gradient và thống kê Dying ReLU.
- `visualization.py`: Các script sử dụng Matplotlib/Seaborn hỗ trợ vẽ và tự động lưu biểu đồ đánh giá.
- `main.py`: File thực thi chính kết nối toàn bộ các module.

## Hướng dẫn sử dụng
1. Cài đặt các thư viện yêu cầu:
   ```bash
   pip install -r requirements.txt
   ```
2. Chạy file thực thi chính:
   ```bash
   python main.py
   ```

*Lưu ý: Sau khi chạy thành công, đồ thị biểu diễn kết quả sẽ được tự động xuất ra thư mục `outputs/` để đính kèm vào báo cáo Word.*

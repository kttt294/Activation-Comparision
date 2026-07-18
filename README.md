# So sánh ảnh hưởng của hàm kích hoạt đến quá trình huấn luyện

Đây là mã nguồn cho bài tập lớn môn **Nhập môn Trí tuệ Nhân tạo**. Dự án được cấu trúc bài bản theo dạng Module Python chuyên nghiệp, nhằm tối ưu hóa tính linh hoạt, khả năng tái sử dụng mã và dễ dàng kiểm soát luồng dữ liệu.

Phiên bản Notebook của dự án trên Google Colab tại đây: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Viic6NWl_HSJAkP8Xg3HylM8cVvRF4sf)

## Cấu trúc dự án

- `config.py`: File cấu hình tập trung. Chứa các tham số huấn luyện cốt lõi (Epochs, Batch size, LR) để quản lý cấu hình tập trung.
- `data_loader.py`: Đóng gói logic tải dữ liệu MNIST và tự động xử lý chia tập Train/Validation.
- `models.py`: Khởi tạo khối mạng Multi-Layer Perceptron (MLP) và định nghĩa các hàm kích hoạt.
- `engine.py`: Chứa vòng lặp huấn luyện, đánh giá mô hình, thu thập số liệu gradient và thống kê Dying ReLU.
- `visualization.py`: Các script sử dụng Matplotlib/Seaborn hỗ trợ vẽ đồ thị.
- `main.py`: File thực thi chính kết nối toàn bộ các module.

## Hướng dẫn cài đặt và sử dụng

1. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```
2. Chạy file thực thi chính để bắt đầu quá trình huấn luyện mô hình:
   ```bash
   python main.py
   ```

*Lưu ý: Sau khi tiến trình hoàn tất, các biểu đồ phân tích kết quả sẽ được tự động kết xuất (export) ra thư mục `outputs/`.*

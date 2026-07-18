# So sánh ảnh hưởng của hàm kích hoạt đến quá trình huấn luyện

Đây là mã nguồn cho bài tập lớn môn **Nhập môn Trí tuệ Nhân tạo**. Dự án tập trung vào việc nghiên cứu, cài đặt và phân tích ảnh hưởng của 4 hàm kích hoạt phổ biến: **Sigmoid, Tanh, ReLU, và Leaky ReLU** đối với hiệu suất và độ ổn định của mạng Multi-Layer Perceptron (MLP) khi huấn luyện trên tập dữ liệu hình ảnh **MNIST**.

Phiên bản Notebook của dự án trên Google Colab tại đây: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)
](https://colab.research.google.com/drive/1Viic6NWl_HSJAkP8Xg3HylM8cVvRF4sf)

---

## Mục lục

1. [Tổng quan dự án](#-tổng-quan-dự-án)
2. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
3. [Môi trường và Cài đặt](#%EF%B8%8F-môi-trường-và-cài-đặt)
4. [Hướng dẫn chạy thực nghiệm](#-hướng-dẫn-chạy-thực-nghiệm)
5. [Tóm tắt các thực nghiệm cốt lõi](#-tóm-tắt-các-thực-nghiệm-cốt-lõi)

---

## Tổng quan dự án

Dự án này không chỉ đơn thuần là phân loại chữ số viết tay, mà đi sâu vào việc giải phẫu **"hộp đen"** của mạng nơ-ron thông qua các góc nhìn toán học và thực nghiệm:

- Đánh giá tốc độ hội tụ (Convergence Rate) và Độ chính xác (Accuracy) của từng hàm kích hoạt trên mạng nông (Shallow Network).
- Phân tích trực quan hiện tượng **Dying ReLU** bằng cách ép Learning Rate lên cao.
- Chứng minh và trực quan hóa hiện tượng **Vanishing Gradient** (Suy giảm đạo hàm) trên mạng sâu (Deep Network) 8 lớp ẩn thông qua biểu đồ cường độ Gradient.

Toàn bộ mã nguồn được thiết kế theo mô hình **Hướng đối tượng (OOP)** và phân tách thành các **Module độc lập** tuân thủ nguyên tắc Kỹ thuật Phần mềm (Software Engineering).

---

## Cấu trúc thư mục

```text
.
├── config.py           # Cấu hình siêu tham số (Hyperparameters), Seed và Device.
├── data_loader.py      # Quản lý dữ liệu: Tải MNIST, chia tập Train/Validation, tạo DataLoader.
├── models.py           # Định nghĩa kiến trúc mạng MLP và khởi tạo trọng số (He/Xavier).
├── engine.py           # Chứa các vòng lặp huấn luyện, đánh giá và logic thu thập Gradient.
├── visualization.py    # Xử lý logic vẽ đồ thị (Loss, Accuracy, Gradient) bằng Matplotlib/Seaborn.
├── main.py             # File thực thi chính, tự động kích hoạt tuần tự các bài thực nghiệm.
└── requirements.txt    # Danh sách các thư viện phụ thuộc.
```

---

## Môi trường và Cài đặt

Yêu cầu hệ thống:

- Hệ điều hành: Windows, macOS, hoặc Linux.
- Python version >= 3.8

**Bước 1:** Khởi tạo môi trường ảo (Khuyến nghị)

```bash
python -m venv venv
# Active trên Windows:
venv\Scripts\activate
# Active trên macOS/Linux:
source venv/bin/activate
```

**Bước 2:** Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

---

## Hướng dẫn chạy thực nghiệm

Để kích hoạt quá trình huấn luyện và đánh giá, chạy tệp lệnh duy nhất:

```bash
python main.py
```

**Luồng thực thi tự động của hệ thống:**

1. **Kiểm tra phần cứng:** Tự động phát hiện và ánh xạ lên GPU (`CUDA`) nếu có, ngược lại sẽ sử dụng CPU.
2. **Tải dữ liệu:** Tự động tải tập MNIST (nếu chưa có) vào thư mục `./data`.
3. **Thực thi mô hình:** Tiến hành huấn luyện song song các hàm kích hoạt theo 3 kịch bản: Mạng Nông, Dying ReLU, và Mạng Sâu.
4. **Kết xuất báo cáo (Export):** Tất cả kết quả đánh giá (Log) sẽ in ra Terminal, đồng thời các **biểu đồ phân tích chuyên sâu sẽ được tự động lưu vào thư mục `outputs/`**.

---

## Tóm tắt các thực nghiệm cốt lõi

Khi chạy lệnh trên, dự án sẽ tự động sinh ra 5 biểu đồ trực quan (được lưu tại `outputs/`):

1. `fig_01_shallow_loss.png` / `fig_02_shallow_acc.png`: So sánh tốc độ học của Sigmoid, Tanh, ReLU trên kiến trúc mạng 3 lớp ẩn.
2. `fig_03_dying_loss.png`: Trực quan hóa sự sụp đổ của mạng khi ReLU bị "chết" ở môi trường Learning Rate lớn (0.1), so sánh với sức chịu đựng của Leaky ReLU.
3. `fig_04_vanishing_grad.png`: Biểu đồ hình vĩ cầm (Violin Plot) chụp X-quang dòng chảy của Gradient qua 8 lớp ẩn, minh chứng toán học sống động cho sự bốc hơi tín hiệu của Sigmoid/Tanh.
4. `fig_05_gradient_evo.png`: Theo dõi quỹ đạo sống sót của Gradient ở lớp đầu tiên trong suốt 50 epochs.

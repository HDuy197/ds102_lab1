### Bài 1: Hồi quy tuyến tính sử dụng NumPy

Mục tiêu:

 -Xây dựng mô hình hồi quy tuyến tính không sử dụng thư viện ngoài (chỉ dùng NumPy).
 
 -Huấn luyện mô hình với phương trình tối ưu hóa θ (hệ số hồi quy) theo công thức:   
 
$$ \theta = (X^T X)^{-1} X^T y $$

Các bước thực hiện:

Tiền xử lý dữ liệu

        Chuyển đổi cột month và day thành dạng số.

        Loại bỏ cột area để tạo tập đặc trưng (X) và tập nhãn (y).

Chia tập dữ liệu

        80% dữ liệu làm tập huấn luyện, 20% làm tập kiểm tra.

Thêm bias vào tập dữ liệu

        Thêm một cột giá trị 1 vào X để biểu diễn hệ số chặn (intercept).

Tính toán hệ số hồi quy θ

        Áp dụng công thức tính θ bằng cách sử dụng NumPy.
Dự đoán và đánh giá mô hình

        Tính MSE và MAE trên tập train và test.
### Bài 2: Chuẩn hóa dữ liệu
Mục tiêu:

Chuẩn hóa dữ liệu về phân phối có kỳ vọng 0 và phương sai 1.

So sánh hiệu suất mô hình trước và sau khi chuẩn hóa.

Các bước thực hiện:

Chuẩn hóa đặc trưng

        Sử dụng công thức chuẩn hóa Standardization:
$$ X_{new} = \frac{X - \mu}{\sigma} $$

        Áp dụng cho cả X_train và X_test.

Chuẩn hóa nhãn (target variable)

        Dữ liệu y (area) có phân phối lệch, do đó cần chuẩn hóa.

Huấn luyện lại mô hình

        Tính toán lại θ sau khi chuẩn hóa.

Chuyển đổi kết quả về giá trị gốc

        Dự đoán trên tập train và test, sau đó đưa kết quả về giá trị ban đầu.
        
Đánh giá mô hình

        So sánh MSE, MAE với kết quả của Bài 1.
### Bài 3: Phát hiện và loại bỏ thuộc tính tương quan cao (Collinearity)
Mục tiêu:

-Loại bỏ các thuộc tính có tương quan cao (có thể gây nhiễu cho mô hình).

-Huấn luyện lại mô hình với tập dữ liệu đã giảm bớt số chiều.

Các bước thực hiện:

Tính toán ma trận tương quan của các đặc trưng

        Sử dụng ma trận tương quan Pearson để đo mức độ tương quan giữa các biến.

Lọc ra các thuộc tính có tương quan cao

        Chỉ giữ lại các thuộc tính có tương quan thấp hơn ngưỡng 0.85.

Loại bỏ thuộc tính dư thừa và huấn luyện lại mô hình

        Huấn luyện lại mô hình với tập dữ liệu đã giảm chiều.
        
Đánh giá hiệu suất mô hình

        Tính toán MSE và MAE
### Bài 4: Hồi quy tuyến tính sử dụng thư viện Scikit-Learn
Mục tiêu:
-Sử dụng thư viện Scikit-Learn để triển khai mô hình hồi quy tuyến tính.
-So sánh với mô hình thủ công đã triển khai trước đó.
Các bước thực hiện:

Huấn luyện mô hình Linear Regression với Scikit-Learn

        Sử dụng LinearRegression() từ thư viện sklearn.
        
Dự đoán kết quả trên tập train và test

        Dự đoán diện tích cháy rừng.

Tính toán lỗi MSE và MAE

        Tính toán MSE và MAE

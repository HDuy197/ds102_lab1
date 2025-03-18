import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Đọc dữ liệu
df = pd.read_csv(r"C:\Users\duyth\Downloads\forestfires.csv")

# Chuyển đổi month & day sang dạng số
df["month"] = df["month"].map({
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
})
df["day"] = df["day"].map({
    "sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6
})

# Kiểm tra dữ liệu sau khi chuyển đổi
print("Dữ liệu sau khi chuyển đổi month và day:")
print(df.head())

# Bài 1

# 4. Tách X và y
X = df.drop(columns=["area"])
y = df["area"]

# Chia tập train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Thêm cột bias (1) vào X_train & X_test
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Tính toán tham số theta theo công thức hồi quy tuyến tính
theta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train

# Dự đoán trên tập train và test
y_train_pred = X_train @ theta
y_test_pred = X_test @ theta

# Tính lỗi MSE và MAE
mse_train = np.mean((y_train_pred - y_train) ** 2)
mse_test = np.mean((y_test_pred - y_test) ** 2)
mae_train = np.mean(np.abs(y_train_pred - y_train))
mae_test = np.mean(np.abs(y_test_pred - y_test))

# In ra kết quả
print("\nKẾT QUẢ BÀI 1 (Trước Chuẩn Hóa)")
print("Theta:", theta)
print("MSE trên tập train:", mse_train)
print("MSE trên tập test:", mse_test)
print("MAE trên tập train:", mae_train)
print("MAE trên tập test:", mae_test)

# Bài 2

# Chuẩn hóa dữ liệu (Standardization)
mean_X = X_train.mean(axis=0)
std_X = X_train.std(axis=0)
std_X[std_X == 0] = 1  # Tránh chia cho 0

X_train_standardized = (X_train - mean_X) / std_X
X_test_standardized = (X_test - mean_X) / std_X

mean_y = y_train.mean()
std_y = y_train.std()

y_train_standardized = (y_train - mean_y) / std_y
y_test_standardized = (y_test - mean_y) / std_y

# Thêm cột bias (1) vào X_train & X_test (chuẩn hóa)
X_train_standardized = np.c_[np.ones(X_train_standardized.shape[0]), X_train_standardized]
X_test_standardized = np.c_[np.ones(X_test_standardized.shape[0]), X_test_standardized]

# Tính toán tham số theta sau khi chuẩn hóa
theta_standardized = np.linalg.pinv(X_train_standardized.T @ X_train_standardized) @ X_train_standardized.T @ y_train_standardized

# Dự đoán trên tập train và test
y_train_pred_standardized = X_train_standardized @ theta_standardized
y_test_pred_standardized = X_test_standardized @ theta_standardized

# Chuyển dự đoán về giá trị gốc
y_train_pred_original = y_train_pred_standardized * std_y + mean_y
y_test_pred_original = y_test_pred_standardized * std_y + mean_y

# Tính lỗi MSE và MAE sau chuẩn hóa
mse_train_standardized = np.mean((y_train_pred_original - y_train) ** 2)
mse_test_standardized = np.mean((y_test_pred_original - y_test) ** 2)
mae_train_standardized = np.mean(np.abs(y_train_pred_original - y_train))
mae_test_standardized = np.mean(np.abs(y_test_pred_original - y_test))

# In ra kết quả
print("\nKẾT QUẢ BÀI 2 (Sau Chuẩn Hóa)")
print("Theta trước chuẩn hóa:", theta)
print("Theta sau chuẩn hóa:", theta_standardized)
print("MSE trên tập train:", mse_train_standardized)
print("MSE trên tập test:", mse_test_standardized)
print("MAE trên tập train:", mae_train_standardized)
print("MAE trên tập test:", mae_test_standardized)

# Bài 3

# Tính ma trận tương quan của dữ liệu gốc (không chuẩn hóa)
correlation_matrix = X.corr()

# Tìm các thuộc tính có tương quan cao hơn ngưỡng
correlation_threshold = 0.85  # Có thể giảm xuống 0.8 để kiểm tra thêm
highly_correlated_features = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            highly_correlated_features.add(correlation_matrix.columns[j])



# Loại bỏ thuộc tính collinear và huấn luyện lại mô hình
X_filtered = X.drop(columns=highly_correlated_features)
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

# Thêm cột bias (1) vào X_train_filtered & X_test_filtered
X_train_filtered = np.c_[np.ones(X_train_filtered.shape[0]), X_train_filtered]
X_test_filtered = np.c_[np.ones(X_test_filtered.shape[0]), X_test_filtered]

# Huấn luyện lại mô hình hồi quy tuyến tính
theta_filtered = np.linalg.pinv(X_train_filtered.T @ X_train_filtered) @ X_train_filtered.T @ y_train_filtered

# Dự đoán trên tập train và test
y_train_pred_filtered = X_train_filtered @ theta_filtered
y_test_pred_filtered = X_test_filtered @ theta_filtered

# Tính lại MSE và MAE
mse_train_filtered = np.mean((y_train_pred_filtered - y_train_filtered) ** 2)
mse_test_filtered = np.mean((y_test_pred_filtered - y_test_filtered) ** 2)
mae_train_filtered = np.mean(np.abs(y_train_pred_filtered - y_train_filtered))
mae_test_filtered = np.mean(np.abs(y_test_pred_filtered - y_test_filtered))

# In ra kết quả
print("\nKẾT QUẢ BÀI 3 (Sau khi loại bỏ collinearity)")
# Hiển thị thuộc tính bị loại bỏ
print("Các thuộc tính bị loại bỏ do collinearity:", highly_correlated_features if highly_correlated_features else "Không có")
print("Theta:", theta_filtered)
print("MSE trên tập train:", mse_train_filtered)
print("MSE trên tập test:", mse_test_filtered)
print("MAE trên tập train:", mae_train_filtered)
print("MAE trên tập test:", mae_test_filtered)

# Hiển thị Heatmap để kiểm tra trực quan collinearity
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Ma Trận Tương Quan")
plt.show(block=True)

# Bài 4

# Khởi tạo mô hình Linear Regression
model = LinearRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập train và test
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Tính lỗi MSE và MAE
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

# In ra kết quả
print("\nKẾT QUẢ BÀI 4 (Linear Regression - Sklearn)")
np.set_printoptions(suppress=True, precision=6)  # Tắt ký hiệu khoa học, làm tròn 6 chữ số
print("Theta:", theta)# Gồm cả bias
print("MSE trên tập train:", mse_train)
print("MSE trên tập test:", mse_test)
print("MAE trên tập train:", mae_train)
print("MAE trên tập test:", mae_test)

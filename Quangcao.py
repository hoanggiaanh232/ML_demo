# Import các thư viện cần thiết
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Đọc dữ liệu từ file CSV
file_path = 'đường dẫn_tới_file/(T5)advertising.csv'  # Cập nhật đường dẫn tới file của bạn
data = pd.read_csv(file_path)

# Chia dữ liệu thành các biến đầu vào (X) và biến mục tiêu (y)
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình với tập huấn luyện
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá hiệu suất của mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R² Score: {r2}')

# In ra các hệ số của mô hình
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)

# Dự đoán doanh số bán hàng cho một ngân sách quảng cáo mới
new_data = pd.DataFrame({
    'TV': [100],
    'Radio': [25],
    'Newspaper': [20]
})
predicted_sales = model.predict(new_data)
print(f'Predicted Sales: {predicted_sales[0]}')

#su dung streamlit
n_tv = int(st. text_input ('Input TV', 100))
n_radio = int(st. text_input ('Input Radio: ', 50))
n_newspaper = int(st.text_input ('Input Newspaper:', 50))
if st.button( 'Predict'):
    new_data = pd.DataFrame({
        'TV': [n_tv],
        'Radio': [n_radio],
        'Newspaper': [n_newspaper]
    })
    predicted_sales = model.predict (new_data)
    st.write(f'Predicted Sales: {predicted_sales[0]}')

#streamlit
n_tv = int(st. text_input ('Input TV', 100))
n_radio = int(st. text_input ('Input Radio: ', 50))
n_newspaper = int(st.text_input ('Input Newspaper:', 50))
if st.button( 'Predict'):
    new_data = pd.DataFrame({
        'TV': [n_tv],
        'Radio': [n_radio],
        'Newspaper': [n_newspaper]
    })
    predicted_sales = model.predict(new_data)
    st.write(f'Predicted Sales: (predicted_sales[0])')
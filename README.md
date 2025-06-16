
# BoMGene: Integrating Boruta–mRMR feature selection for enhanced Gene expression classification
  _Bich-Chung Phan, Thanh Ma, Huu-Hoa Nguyen, and Thanh-Nghi Do_
  
**_ABSTRACT_ **_Lựa chọn đặc trưng là bước quan trọng trong phân tích dữ liệu biểu hiện gen, giúp cải thiện hiệu suất phân loại và giảm chi phí tính toán khi xử lý các tập dữ liệu có số chiều cao. Bài báo này đề xuất phương pháp “lai” (hybrid), được gọi là BoMGene, tích hợp hiệu quả hai kỹ thuật lựa chọn đặc trưng phổ biến là Boruta và Minimum Redundancy Maximum Relevance (mRMR)  nhằm tối ưu hóa không gian đặc trưng và nâng cao chất lượng phân lớp. Thực nghiệm được tiến hành trên 25 bộ dữ liệu biểu hiện gen công khai, sử dụng các thuật toán phân loại phổ biến gồm Support Vector Machine (SVM), Random Forest (RF), XGBoost (XGB) và Gradient Boosting Machine (GBM). Kết quả cho thấy phương pháp kết hợp Boruta–mRMR giúp giảm đáng kể số lượng đặc trưng so với chỉ dùng mRMR, đồng thời cải thiện đáng kể thời gian huấn luyện mà vẫn duy trì hoặc nâng cao độ chính xác phân loại so với các phương pháp chọn đặc trưng đơn lẻ. Phương pháp đề xuất thể hiện rõ ưu thế về độ chính xác, tính ổn định và khả năng ứng dụng trong phân tích dữ liệu biểu hiện gen đa lớp._

Some of results 

1. Comparison charts of accuracy between feature selection methods and experimental algorithms
   
![image](https://github.com/user-attachments/assets/77c0dc4b-34e4-442d-b682-29c2d4bf267d)


2. Comparison charts of training time between feature selection methods and experimental algorithms
   
![image](https://github.com/user-attachments/assets/40cbd466-c8d1-4614-9741-3ac676087445)


3. Number of selection feature with our exprimination

   ![image](https://github.com/user-attachments/assets/d298c815-4d1e-425b-9fa6-dcd089388347)

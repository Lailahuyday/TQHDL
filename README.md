ecommerce-visualization/
│
├── data/
│   ├── raw/                 # Chứa file data gốc (CSV/Excel)
│   └── processed/           # Chứa file data đã làm sạch
│
├── docs/
│   └── requirements.md      # Nội dung đồ án (Copy từ chat vào đây)
│
├── notebooks/               # Chứa file Jupyter Notebook để EDA
│   └── 01_eda.ipynb
│
├── src/                     # Chứa code Python (nếu dùng Streamlit)
│   ├── __init__.py
│   ├── data_processor.py    # Code làm sạch
│   └── app.py               # Code giao diện dashboard
│
├── requirements.txt         # Danh sách thư viện (pandas, plotly, streamlit...)
└── README.md

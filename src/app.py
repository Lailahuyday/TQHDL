"""Dashboard Streamlit drill-down Category → Brand → Product."""

from __future__ import annotations

import colorsys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "cleaned_data.csv"
REQUIRED_COLUMNS = {
    "category_id",
    "brand",
    "name",
    "price",
    "rating_average",
    "quantity_sold",
}


def hsl_to_hex(hue: float, saturation: float, lightness: float) -> str:
    """Tạo màu xanh từ HSL và chuyển sang HEX để Plotly hiển thị ổn định."""
    red, green, blue = colorsys.hls_to_rgb(
        (hue % 360) / 360,
        lightness,
        saturation,
    )
    return "#{:02X}{:02X}{:02X}".format(
        round(red * 255), round(green * 255), round(blue * 255)
    )


def blue_palette(count: int) -> list[str]:
    """Sinh dải màu xanh dương đồng nhất theo hue HSL."""
    if count <= 0:
        return []
    if count == 1:
        lightness_values = [0.48]
    else:
        lightness_values = [0.34 + i * (0.48 / (count - 1)) for i in range(count)]
    return [hsl_to_hex(210, 0.76, lightness) for lightness in lightness_values]


def format_vnd(value: float | int | None, decimals: int = 0) -> str:
    """Định dạng số tiền Việt Nam; trả dấu gạch ngang khi dữ liệu trống."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f} ₫"


def iqr_outliers(values: pd.Series) -> pd.Series:
    """Trả mask cho giá trị ngoài ngưỡng 1.5×IQR; cần tối thiểu 4 giá trị."""
    valid = values.dropna()
    mask = pd.Series(False, index=values.index)
    if len(valid) < 4 or valid.nunique() < 2:
        return mask
    q1, q3 = valid.quantile([0.25, 0.75])
    spread = q3 - q1
    lower = q1 - 1.5 * spread
    upper = q3 + 1.5 * spread
    mask.loc[valid.index] = (valid < lower) | (valid > upper)
    return mask


@st.cache_data(show_spinner="Đang tải dữ liệu sản phẩm...")
def load_data(path: str) -> pd.DataFrame:
    """Đọc CSV, giữ category ID dạng chuỗi và chuẩn hóa trường phân tích."""
    dataframe = pd.read_csv(
        path,
        dtype={
            "category_id": "string",
            "source_category_id": "string",
            "id": "string",
            "sku": "string",
            "brand": "string",
        },
    )
    missing_columns = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing_columns:
        raise ValueError("CSV thiếu cột bắt buộc: " + ", ".join(missing_columns))

    for column in ("price", "list_price", "rating_average", "review_count", "quantity_sold"):
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe["category_id"] = dataframe["category_id"].str.strip()
    dataframe["brand"] = dataframe["brand"].str.strip()
    dataframe["brand"] = dataframe["brand"].replace("", pd.NA).fillna("Unknown")
    dataframe["rating_average"] = dataframe["rating_average"].fillna(0)

    if "revenue_est" not in dataframe.columns:
        dataframe["revenue_est"] = dataframe["price"] * dataframe["quantity_sold"]
    else:
        dataframe["revenue_est"] = pd.to_numeric(
            dataframe["revenue_est"], errors="coerce"
        )
    return dataframe


st.set_page_config(
    page_title="Phân tích danh mục sản phẩm",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #f4f9ff, #e8f2ff);
        border: 1px solid #d7e8fb;
        padding: 16px 18px;
        border-radius: 14px;
      }
      [data-testid="stMetricLabel"] { color: #315b83; }
      .block-container { padding-top: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Dashboard phân tích danh mục sản phẩm")
st.caption(
    "Khám phá dữ liệu theo ba tầng: Danh mục → Thương hiệu → Sản phẩm. "
    "Doanh thu là giá bán nhân lượng bán, dùng như một ước tính."
)

if not DATA_PATH.is_file():
    st.error(f"Không tìm thấy dữ liệu: {DATA_PATH}")
    st.info(
        "Hãy chạy bước làm sạch để tạo data/processed/cleaned_data.csv "
        "trước khi mở dashboard."
    )
    st.stop()

try:
    data = load_data(str(DATA_PATH))
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Không thể đọc dữ liệu: {error}")
    st.stop()

data = data.dropna(subset=["category_id"])
if data.empty:
    st.warning("Dữ liệu không có category_id hợp lệ để phân tích.")
    st.stop()

category_summary = (
    data.groupby("category_id", observed=True)
    .agg(products=("name", "size"), revenue=("revenue_est", "sum"))
    .sort_values(["revenue", "products"], ascending=False)
)
category_ids = category_summary.index.astype(str).tolist()
category_counts = category_summary["products"].to_dict()

st.sidebar.title("Bộ lọc drill-down")
st.sidebar.caption("Chọn danh mục trước, sau đó chọn thương hiệu.")
category_id = st.sidebar.selectbox(
    "Tầng 1 · Danh mục (category_id)",
    options=category_ids,
    format_func=lambda value: f"{value} · {category_counts.get(value, 0):,} SP",
    key="category_id",
)

category_data = data.loc[data["category_id"] == category_id].copy()
brand_summary = (
    category_data.groupby("brand", observed=True)
    .agg(revenue=("revenue_est", "sum"), products=("name", "size"))
    .sort_values(["revenue", "products"], ascending=False)
)

known_brand_summary = brand_summary.drop(index="Unknown", errors="ignore")
ranked_brands = known_brand_summary.index.astype(str).tolist()
if "Unknown" in brand_summary.index:
    ranked_brands.append("Unknown")

if ranked_brands:
    selected_brand = st.sidebar.selectbox(
        "Tầng 2 · Thương hiệu",
        options=ranked_brands,
        format_func=lambda value: "Unknown · Chưa rõ thương hiệu" if value == "Unknown" else value,
        key=f"brand_for_{category_id}",
    )
else:
    selected_brand = None
    st.sidebar.info("Danh mục này chưa có thương hiệu để chọn.")

# Tầng 1: KPI của danh mục đang chọn.
st.header(f"Tầng 1 · Danh mục {category_id}")
total_revenue = category_data["revenue_est"].sum(min_count=1)
average_price = category_data["price"].mean()
average_rating = category_data["rating_average"].mean()
product_count = len(category_data)

kpi_columns = st.columns(4, gap="medium")
kpi_columns[0].metric("Tổng doanh thu ước tính", format_vnd(total_revenue))
kpi_columns[1].metric("Giá trung bình", format_vnd(average_price))
kpi_columns[2].metric(
    "Rating trung bình",
    "—" if pd.isna(average_rating) else f"{average_rating:.2f} / 5",
)
kpi_columns[3].metric("Tổng sản phẩm", f"{product_count:,}")
st.caption(
    f"Danh mục này có {len(known_brand_summary):,} thương hiệu đã biết; "
    "nhóm Unknown không được tính là một thương hiệu."
)

# Tầng 2: xếp hạng thương hiệu trong danh mục đang chọn.
st.divider()
st.header("Tầng 2 · So sánh thương hiệu")
chart_column, insight_column = st.columns([1.65, 1], gap="large")

top10 = known_brand_summary.head(10).sort_values("revenue", ascending=True)
with chart_column:
    if top10.empty:
        st.info("Danh mục chưa có thương hiệu đã biết để xếp hạng.")
    else:
        brand_chart = px.bar(
            top10.reset_index(),
            x="revenue",
            y="brand",
            orientation="h",
            color="brand",
            color_discrete_sequence=blue_palette(len(top10)),
            labels={"revenue": "Doanh thu ước tính (₫)", "brand": "Thương hiệu"},
            title="Top 10 thương hiệu theo doanh thu",
            custom_data=["products"],
        )
        brand_chart.update_traces(
            texttemplate="%{x:,.0f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>Doanh thu: %{x:,.0f} ₫<br>"
                "Số sản phẩm: %{customdata[0]:,}<extra></extra>"
            ),
        )
        brand_chart.update_layout(
            template="plotly_white",
            height=max(430, 34 * len(top10) + 150),
            showlegend=False,
            margin={"l": 30, "r": 50, "t": 70, "b": 55},
            yaxis={"categoryorder": "array", "categoryarray": top10.index.tolist()},
        )
        brand_chart.update_xaxes(tickformat=",.0f", gridcolor="#e4effa")
        st.plotly_chart(brand_chart, use_container_width=True)

with insight_column:
    st.subheader("Tóm tắt thương hiệu")
    st.metric("Thương hiệu có doanh thu", f"{len(known_brand_summary):,}")
    if not known_brand_summary.empty:
        leader = known_brand_summary.sort_values("revenue", ascending=False).iloc[0]
        st.metric(
            "Dẫn đầu doanh thu",
            str(known_brand_summary.index[0]),
            help=f"Doanh thu ước tính: {format_vnd(leader['revenue'])}",
        )
    unknown_products = int(brand_summary.loc["Unknown", "products"]) if "Unknown" in brand_summary.index else 0
    st.metric("Sản phẩm chưa rõ brand", f"{unknown_products:,}")
    st.caption("Biểu đồ xếp hạng loại nhóm Unknown khỏi danh sách thương hiệu.")

# Tầng 3: thông tin sản phẩm của thương hiệu được chọn.
st.divider()
if selected_brand is not None:
    brand_data = category_data.loc[category_data["brand"] == selected_brand].copy()
    brand_title = "Chưa rõ thương hiệu" if selected_brand == "Unknown" else selected_brand
    st.header(f"Tầng 3 · Sản phẩm thuộc {brand_title}")
    st.caption(f"{len(brand_data):,} sản phẩm trong danh mục {category_id}.")

    table_column, scatter_column = st.columns([1.1, 1], gap="large")
    table_columns = ["name", "price", "rating_average", "quantity_sold"]
    if "revenue_est" in brand_data.columns:
        brand_data = brand_data.sort_values("revenue_est", ascending=False, na_position="last")

    with table_column:
        st.subheader("Danh sách sản phẩm")
        st.dataframe(
            brand_data[table_columns],
            column_config={
                "name": st.column_config.TextColumn("Sản phẩm", width="large"),
                "price": st.column_config.NumberColumn("Giá", format="%,.0f ₫"),
                "rating_average": st.column_config.NumberColumn("Rating", format="%.2f"),
                "quantity_sold": st.column_config.NumberColumn("Đã bán", format="%,.0f"),
            },
            hide_index=True,
            use_container_width=True,
            height=520,
        )

    with scatter_column:
        st.subheader("Giá và lượng bán")
        scatter_data = brand_data.dropna(subset=["price", "quantity_sold"]).copy()
        if scatter_data.empty:
            st.info("Không có sản phẩm đủ dữ liệu giá và lượng bán để vẽ scatter plot.")
        else:
            outlier_mask = iqr_outliers(scatter_data["price"]) | iqr_outliers(
                scatter_data["quantity_sold"]
            )
            scatter_data["Tình trạng"] = "Trong ngưỡng IQR"
            scatter_data.loc[outlier_mask, "Tình trạng"] = "Ứng viên bất thường"
            scatter = px.scatter(
                scatter_data,
                x="price",
                y="quantity_sold",
                color="Tình trạng",
                color_discrete_map={
                    "Trong ngưỡng IQR": hsl_to_hex(210, 0.76, 0.48),
                    "Ứng viên bất thường": "#E45756",
                },
                hover_name="name",
                hover_data={
                    "price": ":,.0f",
                    "quantity_sold": ":,.0f",
                    "rating_average": ":.2f",
                    "Tình trạng": True,
                },
                render_mode="webgl",
                opacity=0.78,
                title="Scatter plot · Ứng viên ngoài ngưỡng 1.5×IQR được tô đỏ",
                labels={
                    "price": "Giá (₫)",
                    "quantity_sold": "Số lượng đã bán",
                    "Tình trạng": "Phân loại",
                },
            )
            scatter.update_layout(
                template="plotly_white",
                height=520,
                margin={"l": 70, "r": 20, "t": 75, "b": 65},
                legend={"orientation": "h", "y": 1.08, "x": 0},
            )
            scatter.update_xaxes(tickformat=",.0f", gridcolor="#e4effa")
            scatter.update_yaxes(tickformat=",.0f", gridcolor="#e4effa")
            st.plotly_chart(scatter, use_container_width=True)

st.caption(
    "Nguồn: data/processed/cleaned_data.csv · Doanh thu ước tính = price × quantity_sold. "
    "Các điểm đỏ là ứng viên ngoại lệ theo quy tắc 1.5×IQR, cần được kiểm tra theo ngữ cảnh."
)

"""EDA danh mục và thương hiệu từ cleaned_data.csv.

Ví dụ chạy từ thư mục gốc repository:
    python notebooks/02_category_brand_analysis.py
    python notebooks/02_category_brand_analysis.py --category-id 2514
    python notebooks/02_category_brand_analysis.py --category-id 2514 --all-brands

Nếu thiếu Plotly, cài bằng: python -m pip install plotly
"""

from __future__ import annotations

import argparse
import colorsys
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "cleaned_data.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "category_brand_analysis"
REQUIRED_COLUMNS = {
    "category_id",
    "brand",
    "price",
    "rating_average",
    "revenue_est",
}


def hsl_to_hex(hue: float, saturation: float, lightness: float) -> str:
    """Tạo màu xanh từ HSL rồi chuyển sang HEX để Plotly hiển thị ổn định."""
    red, green, blue = colorsys.hls_to_rgb(
        (hue % 360) / 360,
        lightness,
        saturation,
    )
    return "#{:02X}{:02X}{:02X}".format(
        round(red * 255), round(green * 255), round(blue * 255)
    )


def blue_palette(count: int) -> list[str]:
    """Bảng màu đơn sắc xanh dương với độ sáng biến thiên theo HSL."""
    if count <= 0:
        return []
    if count == 1:
        return [hsl_to_hex(210, 0.78, 0.48)]
    lightness_values = [0.34 + i * (0.48 / (count - 1)) for i in range(count)]
    return [hsl_to_hex(210, 0.76, lightness) for lightness in lightness_values]


def load_data(input_path: Path) -> pd.DataFrame:
    """Đọc dữ liệu đã làm sạch và xác thực các trường dùng trong phân tích."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {input_path}. Hãy chạy bước data cleaning trước "
            "hoặc truyền file bằng --input."
        )

    data = pd.read_csv(
        input_path,
        dtype={"category_id": "string", "brand": "string"},
    )
    missing_columns = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing_columns:
        raise ValueError(
            "Dữ liệu thiếu các cột bắt buộc: " + ", ".join(missing_columns)
        )

    for column in ("price", "rating_average", "revenue_est"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["category_id"] = data["category_id"].str.strip()
    data["brand"] = data["brand"].str.strip().replace("", pd.NA).fillna("Unknown")
    return data


def select_default_category(data: pd.DataFrame) -> str:
    """Chọn danh mục đa dạng brand nhất; dùng số sản phẩm để phá hòa."""
    available = data.dropna(subset=["category_id"])
    if available.empty:
        raise ValueError("Không có category_id hợp lệ trong dữ liệu.")

    known_brands = available.loc[available["brand"] != "Unknown"]
    diversity = (
        known_brands.groupby("category_id", observed=True)["brand"]
        .nunique()
        .rename("brand_count")
    )
    product_counts = available.groupby("category_id", observed=True).size().rename("product_count")
    ranking = pd.concat([diversity, product_counts], axis=1).fillna(0)
    ranking = ranking.sort_values(
        ["brand_count", "product_count"], ascending=False
    )
    return str(ranking.index[0])


def validate_category(data: pd.DataFrame, category_id: str) -> pd.DataFrame:
    """Lọc một category; báo các mã hợp lệ nếu mã được chọn không tồn tại."""
    selected = data.loc[data["category_id"] == str(category_id)].copy()
    if selected.empty:
        counts = (
            data.dropna(subset=["category_id"])
            .groupby("category_id", observed=True)
            .size()
            .sort_values(ascending=False)
        )
        examples = ", ".join(counts.head(10).index.astype(str))
        raise ValueError(
            f"Không tìm thấy category_id={category_id}. "
            f"Một số mã có nhiều sản phẩm: {examples}"
        )
    return selected


def build_figures(
    selected: pd.DataFrame,
    category_id: str,
    *,
    max_brands_boxplot: int = 20,
) -> dict[str, object]:
    """Tạo KPI, top brand theo doanh thu, treemap và box plot."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise RuntimeError(
            "Plotly is not installed. Install it with: python -m pip install plotly"
        ) from error

    blue_dark = hsl_to_hex(210, 0.78, 0.22)
    blue_mid = hsl_to_hex(210, 0.76, 0.47)
    blue_light = hsl_to_hex(210, 0.72, 0.86)
    theme = "plotly_white"

    # KPI mô tả toàn bộ sản phẩm trong category; brand Unknown không được tính là thương hiệu.
    known_brand_count = selected.loc[selected["brand"] != "Unknown", "brand"].nunique()
    kpis = [
        ("Tổng sản phẩm", len(selected), ",.0f", ""),
        ("Số thương hiệu", known_brand_count, ",.0f", ""),
        ("Doanh thu ước tính", selected["revenue_est"].sum(), ",.0f", " ₫"),
        ("Giá trung bình", selected["price"].mean(), ",.0f", " ₫"),
        ("Rating trung bình", selected["rating_average"].mean(), ".2f", " / 5"),
    ]
    kpi_figure = make_subplots(
        rows=1,
        cols=len(kpis),
        specs=[[{"type": "indicator"}] * len(kpis)],
        horizontal_spacing=0.045,
    )
    for column, (label, value, number_format, suffix) in enumerate(kpis, start=1):
        kpi_figure.add_trace(
            go.Indicator(
                mode="number",
                value=float(value) if pd.notna(value) else 0,
                title={"text": label, "font": {"size": 13, "color": blue_dark}},
                number={
                    "valueformat": number_format,
                    "suffix": suffix,
                    "font": {"size": 23, "color": blue_dark},
                },
            ),
            row=1,
            col=column,
        )
    kpi_figure.update_layout(
        title={"text": f"Tổng quan danh mục {category_id}", "x": 0.02},
        template=theme,
        height=190,
        margin={"l": 12, "r": 12, "t": 58, "b": 15},
        paper_bgcolor="white",
    )

    known = selected.loc[selected["brand"] != "Unknown"]
    brand_revenue = (
        known.dropna(subset=["revenue_est"])
        .groupby("brand", observed=True)["revenue_est"]
        .sum()
        .sort_values(ascending=False)
    )
    top_revenue = brand_revenue.head(10).sort_values(ascending=True)
    bar_figure = go.Figure(
        go.Bar(
            x=top_revenue.values,
            y=top_revenue.index.astype(str),
            orientation="h",
            marker={"color": blue_palette(len(top_revenue))},
            text=top_revenue.values,
            texttemplate="%{x:,.0f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Doanh thu: %{x:,.0f} ₫<extra></extra>",
        )
    )
    bar_figure.update_layout(
        title=f"Top 10 thương hiệu theo doanh thu · Category {category_id}",
        xaxis_title="Doanh thu ước tính (₫)",
        yaxis_title="Thương hiệu",
        template=theme,
        height=max(430, 30 * len(top_revenue) + 150),
        margin={"l": 145, "r": 60, "t": 75, "b": 70},
        font={"color": blue_dark},
    )
    bar_figure.update_xaxes(tickformat=",.0f", gridcolor=blue_light)

    # Treemap giữ riêng nhóm Unknown để phản ánh phần doanh thu chưa gắn thương hiệu.
    treemap_revenue = (
        selected.dropna(subset=["revenue_est"])
        .groupby("brand", observed=True)["revenue_est"]
        .sum()
        .rename("revenue_est")
        .reset_index()
    )
    treemap_revenue = treemap_revenue.loc[treemap_revenue["revenue_est"] > 0]
    treemap_figure = px.treemap(
        treemap_revenue,
        path=[px.Constant(f"Category {category_id}"), "brand"],
        values="revenue_est",
        color="revenue_est",
        color_continuous_scale=[blue_light, blue_mid, blue_dark],
        custom_data=["revenue_est"],
        title=f"Cơ cấu doanh thu theo thương hiệu · Category {category_id}",
    )
    treemap_figure.update_traces(
        textinfo="label+percent parent",
        hovertemplate="%{label}<br>Doanh thu: %{value:,.0f} ₫<br>"
        "Tỷ trọng trong danh mục: %{percentRoot:.1%}<extra></extra>",
    )
    treemap_figure.update_layout(
        template=theme,
        height=650,
        margin={"l": 12, "r": 12, "t": 72, "b": 12},
        font={"color": blue_dark},
        coloraxis_colorbar={"title": "Doanh thu (₫)", "tickformat": ",.0f"},
    )

    # Box plot dùng tối đa N brand nhiều sản phẩm nhất để nhãn dễ đọc; 0 nghĩa là tất cả.
    box_data = known.dropna(subset=["price"])
    brand_sizes = box_data.groupby("brand", observed=True).size().sort_values(ascending=False)
    if max_brands_boxplot > 0:
        included_brands = brand_sizes.head(max_brands_boxplot).index
        box_data = box_data.loc[box_data["brand"].isin(included_brands)]
    box_brands = sorted(box_data["brand"].dropna().unique().astype(str))
    box_figure = px.box(
        box_data,
        x="brand",
        y="price",
        color="brand",
        category_orders={"brand": box_brands},
        color_discrete_sequence=blue_palette(len(box_brands)),
        points="outliers",
        title=(
            f"Phân phối giá theo thương hiệu · Category {category_id}"
            + (f" · tối đa {max_brands_boxplot} brand theo số sản phẩm"
               if max_brands_boxplot > 0 else " · tất cả brand")
        ),
    )
    box_figure.update_layout(
        template=theme,
        height=650,
        xaxis_title="Thương hiệu",
        yaxis_title="Giá (₫)",
        showlegend=False,
        margin={"l": 80, "r": 25, "t": 75, "b": 150},
        font={"color": blue_dark},
    )
    box_figure.update_xaxes(tickangle=-55, gridcolor=blue_light)
    box_figure.update_yaxes(tickformat=",.0f", gridcolor=blue_light)

    return {
        "kpis": kpi_figure,
        "top_brands": bar_figure,
        "revenue_treemap": treemap_figure,
        "price_boxplot": box_figure,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDA cap Category va Brand bang Pandas va Plotly."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV a lam sach (mac inh: data/processed/cleaned_data.csv).",
    )
    parser.add_argument(
        "--category-id",
        help="Ma category can phan tich; mac inh chon category co nhieu brand nhat.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Thu muc luu cac bieu o HTML tuong tac.",
    )
    parser.add_argument(
        "--max-brands-boxplot",
        type=int,
        default=20,
        help="So brand toi a tren box plot; at 0 e hien thi tat ca.",
    )
    parser.add_argument(
        "--all-brands",
        action="store_true",
        help="Viet tat cua --max-brands-boxplot 0.",
    )
    args = parser.parse_args()

    if args.max_brands_boxplot < 0:
        parser.error("--max-brands-boxplot phai lon hon hoac bang 0.")

    data = load_data(args.input)
    category_id = str(args.category_id) if args.category_id else select_default_category(data)
    selected = validate_category(data, category_id)
    max_brands = 0 if args.all_brands else args.max_brands_boxplot

    figures = build_figures(
        selected,
        category_id,
        max_brands_boxplot=max_brands,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_category_id = re.sub(r"[^A-Za-z0-9_-]", "_", category_id)
    for name, figure in figures.items():
        output_path = args.output_dir / f"category_{safe_category_id}_{name}.html"
        figure.write_html(output_path, include_plotlyjs="cdn", full_html=True)

    print(f"Category {category_id}: {len(selected):,} san pham")
    print(f"a luu {len(figures)} bieu o vao: {args.output_dir}")
    print(
        "Ghi chu: Unknown uoc loai khoi xep hang/box plot; "
        "treemap van the hien doanh thu nhom Unknown."
    )


if __name__ == "__main__":
    main()

"""Đọc, làm sạch và bổ sung biến cho dữ liệu sản phẩm Tiki."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "data.csv"
WORKSPACE_DATA_PATH = REPO_ROOT.parent / "data" / "tiki_all_listings.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "cleaned_data.csv"

REQUIRED_COLUMNS = {
    "source_category_id",
    "id",
    "sku",
    "name",
    "price",
    "list_price",
    "discount_rate",
    "rating_average",
    "review_count",
    "quantity_sold",
    "brand",
    "category_id",
    "thumbnail_url",
    "product_url",
}

NUMERIC_COLUMNS = (
    "price",
    "list_price",
    "discount_rate",
    "rating_average",
    "review_count",
    "quantity_sold",
)


def default_input_path() -> Path:
    """Chọn dữ liệu đầu vào theo cấu trúc chuẩn hoặc dữ liệu sẵn có trong workspace."""
    if RAW_DATA_PATH.is_file():
        return RAW_DATA_PATH
    if WORKSPACE_DATA_PATH.is_file():
        return WORKSPACE_DATA_PATH
    # Giữ đường dẫn chuẩn làm mặc định để báo lỗi cụ thể nếu chưa có dữ liệu.
    return RAW_DATA_PATH


def load_data(input_path: str | Path) -> pd.DataFrame:
    """Đọc CSV và kiểm tra các cột đầu vào cần thiết."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu: {path}. Hãy đặt CSV tại data/raw/data.csv "
            "hoặc truyền đường dẫn bằng tùy chọn --input."
        )

    # Đọc các mã định danh dưới dạng chuỗi để giữ nguyên số 0 ở đầu nếu có.
    dataframe = pd.read_csv(
        path,
        dtype={
            "source_category_id": "string",
            "category_id": "string",
            "id": "string",
            "sku": "string",
        },
    )
    missing_columns = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "CSV thiếu các cột bắt buộc: " + ", ".join(missing_columns)
        )
    return dataframe


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch giá trị, lọc dòng không hợp lệ và thêm các biến phái sinh."""
    result = dataframe.copy()

    # Chuẩn hóa cột số; giá trị không chuyển được thành số sẽ thành NaN.
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    # Bổ sung giá trị thiếu theo yêu cầu; chuỗi rỗng ở brand cũng được xem là thiếu.
    result["rating_average"] = result["rating_average"].fillna(0)
    result["review_count"] = result["review_count"].fillna(0)
    result["brand"] = result["brand"].astype("string").str.strip()
    result["brand"] = result["brand"].replace("", pd.NA).fillna("Unknown")

    # Giữ nguyên dòng có giá/lượng bán bị thiếu để không loại bỏ dữ liệu ngoài yêu cầu.
    valid_rows = ~((result["price"] <= 0) | (result["quantity_sold"] < 0))
    result = result.loc[valid_rows].copy()

    # Các chỉ số doanh thu và giảm giá ước tính dựa trên giá trị đang có trong CSV.
    result["discount_amount"] = result["list_price"] - result["price"]
    result["revenue_est"] = result["price"] * result["quantity_sold"]

    # Các khoảng giá gồm: dưới 100k; 100k–300k; trên 300k–500k; trên 500k.
    price = result["price"]
    result["price_band"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[price < 100_000, "price_band"] = "<100k"
    result.loc[price.between(100_000, 300_000, inclusive="both"), "price_band"] = (
        "100k-300k"
    )
    result.loc[(price > 300_000) & (price <= 500_000), "price_band"] = "300k-500k"
    result.loc[price > 500_000, "price_band"] = ">500k"

    # Điểm 0 (bao gồm sản phẩm chưa có rating) nằm trong nhóm Kém theo các nhãn yêu cầu.
    rating = result["rating_average"]
    result["rating_band"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[rating < 2, "rating_band"] = "Kém"
    result.loc[(rating >= 2) & (rating < 3), "rating_band"] = "TB"
    result.loc[(rating >= 3) & (rating < 4), "rating_band"] = "Tốt"
    result.loc[rating >= 4, "rating_band"] = "Xuất sắc"

    return result


def process_csv(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Chạy toàn bộ quy trình và ghi CSV sạch ra đĩa."""
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"File đầu ra đã tồn tại: {destination}. "
            "Để ghi đè có chủ ý, hãy chạy với --overwrite."
        )

    cleaned = clean_data(load_data(input_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(destination, index=False, encoding="utf-8-sig")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Làm sạch dữ liệu sản phẩm Tiki và tạo các biến phân tích."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_path(),
        help="Đường dẫn CSV đầu vào (mặc định: tự tìm data/raw/data.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Đường dẫn CSV đầu ra (mặc định: data/processed/cleaned_data.csv).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Cho phép ghi đè file đầu ra nếu file đó đã tồn tại.",
    )
    args = parser.parse_args()

    cleaned = process_csv(args.input, args.output, overwrite=args.overwrite)
    # Dùng ký tự ASCII để thông báo vẫn in được trên console Windows không phải UTF-8.
    print(f"Saved {len(cleaned):,} rows to: {args.output}")


if __name__ == "__main__":
    main()

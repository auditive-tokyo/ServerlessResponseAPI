type FilterValue = string | number | boolean;
type FilterType = "eq" | "ne" | "gt" | "gte" | "lt" | "lte";

interface Filter {
    key: string;
    type: FilterType;
    value: FilterValue;
}

/**
 * OpenAI Vector Search用のフィルターキーを作成する関数
 *
 * @param {Array<FilterValue>} values - フィルタリングする値のリスト
 * @param {string} filterType - フィルタリングタイプ (eq, ne, gt, gte, lt, lte)
 * @param {string} filterKey - フィルターキー名（環境変数VECTOR_SEARCH_FILTER_KEYまたは指定値）
 * @returns {Array<Filter>} フィルター辞書のリスト
 *
 * @example
 * createFilterKeys(["201", "common"], "eq", "room_number")
 * // [
 * //   { key: "room_number", type: "eq", value: "201" },
 * //   { key: "room_number", type: "eq", value: "common" }
 * // ]
 *
 * createFilterKeys([100, 200], "gt", "price")
 * // [
 * //   { key: "price", type: "gt", value: 100 },
 * //   { key: "price", type: "gt", value: 200 }
 * // ]
 */
export function createFilterKeys(
    values: FilterValue[],
    filterType: FilterType = "eq",
    filterKey: string
): Filter[] {
    return values.map(value => ({
        key: filterKey,
        type: filterType,
        value: value
    }));
}
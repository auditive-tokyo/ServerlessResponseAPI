from typing import List, Dict, Union, Optional

def create_filter_keys(
    values: List[Union[str, int, bool]],
    filter_type: str = "eq",
    filter_key: Optional[str] = None
) -> List[Dict[str, Union[str, int, bool]]]:
    """
    OpenAI Vector Search用のフィルターキーを作成する関数
    
    Args:
        values: フィルタリングする値のリスト
        filter_type: フィルタリングタイプ (eq, ne, gt, gte, lt, lte)
        filter_key: フィルターキー名（環境変数VECTOR_SEARCH_FILTER_KEYまたは指定値）
    
    Returns:
        フィルター辞書のリスト
    
    Examples:
        >>> create_filter_keys(["201", "common"])
        [
            {"key": "room_number", "type": "eq", "value": "201"},
            {"key": "room_number", "type": "eq", "value": "common"}
        ]
        
        >>> create_filter_keys([100, 200], filter_type="gt", filter_key="price")
        [
            {"key": "price", "type": "gt", "value": 100},
            {"key": "price", "type": "gt", "value": 200}
        ]
    """
    
    # フィルター辞書のリストを作成
    filters = []
    for value in values:
        filter_dict = {
            "key": filter_key,
            "type": filter_type,
            "value": value
        }
        filters.append(filter_dict)
    
    return filters

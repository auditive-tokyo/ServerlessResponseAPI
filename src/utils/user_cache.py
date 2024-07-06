from flask_caching import Cache
from collections import deque
from typing import Dict, Any, List, cast, Optional, Union
import numpy as np

def set_user_cache(
    cache: Cache, 
    user_id: str, 
    user_message: str, 
    history: List[Dict[str, str]], 
    token_limit: int, 
    local_log_option: str, 
    actual_titles: Optional[List[str]] = None, 
    actual_urls: Optional[List[str]] = None, 
    closest_titles: Optional[List[str]] = None, 
    combined_scores: Optional[Union[List[float], np.ndarray]] = None, 
    closest_vector_indices: Optional[Union[List[int], np.ndarray]] = None, 
    matched_ids: Optional[List[str]] = None, 
    cognito_user_id: Optional[str] = None, 
    local_model: Optional[str] = None, 
    local_history_maxlen: Optional[int] = None, 
    headers: Optional[Dict[str, str]] = None
):
    cache.set('user_id', user_id)
    cache.set(f"{user_id}_user_message", user_message)
    cache.set(f"{user_id}_history", list(history))
    cache.set(f"{user_id}_token_limit", token_limit)
    cache.set(f"{user_id}_local_log_option", local_log_option)

    cache.set(f"{user_id}_actual_titles", actual_titles if actual_titles is not None else [])
    cache.set(f"{user_id}_actual_urls", actual_urls if actual_urls is not None else [])
    cache.set(f"{user_id}_closest_titles", closest_titles if closest_titles is not None else [])
    
    if combined_scores is not None:
        cache.set(f"{user_id}_combined_scores", combined_scores.tolist() if isinstance(combined_scores, np.ndarray) else combined_scores)
    else:
        cache.set(f"{user_id}_combined_scores", [])
    
    if closest_vector_indices is not None:
        cache.set(f"{user_id}_closest_vector_indices", closest_vector_indices.tolist() if isinstance(closest_vector_indices, np.ndarray) else closest_vector_indices)
    else:
        cache.set(f"{user_id}_closest_vector_indices", [])
    
    cache.set(f"{user_id}_matched_ids", matched_ids if matched_ids is not None else [])
    cache.set(f"{user_id}_cognito_user_id", cognito_user_id if cognito_user_id is not None else '')
    cache.set(f"{user_id}_local_model", local_model if local_model is not None else '')
    cache.set(f"{user_id}_local_history_maxlen", local_history_maxlen if local_history_maxlen is not None else 0)
    cache.set(f"{user_id}_headers", headers if headers is not None else {})


def get_user_cache(cache, user_id: str) -> Dict[str, Any]:
    cache_data = {key: cache.get(f"{user_id}_{key}") for key in [
        'user_message', 'history', 'token_limit', 'local_log_option',
        'cognito_user_id', 'local_model', 'headers', 'actual_titles',
        'actual_urls', 'closest_titles', 'combined_scores',
        'closest_vector_indices', 'matched_ids', 'local_history_maxlen'
    ]}

    # closest_vector_indicesの処理を修正
    closest_vector_indices = cache_data['closest_vector_indices']
    if isinstance(closest_vector_indices, np.ndarray):
        closest_vector_indices = closest_vector_indices.tolist()
    elif closest_vector_indices is None:
        closest_vector_indices = []

    return {
        'user_id': user_id,
        'user_message': cache_data['user_message'] or '',
        'user_history': deque(cast(List[Dict[str, str]], cache_data['history'] or []), maxlen=int(cache_data['local_history_maxlen'] or 0)),
        'token_limit': cache_data['token_limit'] or 0,
        'local_log_option': cache_data['local_log_option'] or '',
        'cognito_user_id': cache_data['cognito_user_id'] or '',
        'local_model': cache_data['local_model'] or '',
        'headers': cache_data['headers'] or {},
        'actual_titles': cache_data['actual_titles'] or [],
        'actual_urls': cache_data['actual_urls'] or [],
        'closest_titles': cache_data['closest_titles'] or [],
        'combined_scores': cache_data['combined_scores'] or [],
        'closest_vector_indices': closest_vector_indices,
        'matched_ids': cache_data['matched_ids'] or []
    }
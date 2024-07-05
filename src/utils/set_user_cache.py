from flask_caching import Cache
from typing import List, Dict, Any

def set_user_cache(cache: Cache, user_id: str, user_message: str, history: List[Dict[str, str]], token_limit: int, local_log_option: str, actual_titles: List[str], actual_urls: List[str], closest_titles: List[str], combined_scores: List[float], closest_vector_indices: List[int], matched_ids: List[str], cognito_user_id: str, local_model: str, local_history_maxlen: int, headers: Dict[str, str]):
    cache.set('user_id', user_id)
    cache.set(f"{user_id}_user_message", user_message)
    cache.set(f"{user_id}_history", list(history))
    cache.set(f"{user_id}_token_limit", token_limit)
    cache.set(f"{user_id}_local_log_option", local_log_option)

    if actual_titles:
        cache.set(f"{user_id}_actual_titles", actual_titles)
    if actual_urls:
        cache.set(f"{user_id}_actual_urls", actual_urls)

    if closest_titles:
        cache.set(f"{user_id}_closest_titles", closest_titles)
        cache.set(f"{user_id}_combined_scores", combined_scores)
        cache.set(f"{user_id}_closest_vector_indices", closest_vector_indices)

    if matched_ids:
        cache.set(f"{user_id}_matched_ids", matched_ids)

    if cognito_user_id:
        cache.set(f"{user_id}_cognito_user_id", cognito_user_id)

    if local_model:
        cache.set(f"{user_id}_local_model", local_model)

    if local_history_maxlen:
        cache.set(f"{user_id}_local_history_maxlen", local_history_maxlen)

    if headers:
        cache.set(f"{user_id}_headers", headers)
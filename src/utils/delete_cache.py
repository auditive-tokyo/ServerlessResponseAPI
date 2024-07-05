from flask_caching import Cache

def clear_user_cache(cache: Cache, user_id: str):
    if cache.get('user_id'):
        cache.delete('user_id')
    if cache.get(f"{user_id}_user_message"):
        cache.delete(f"{user_id}_user_message")
    if cache.get(f"{user_id}_history"):
        cache.delete(f"{user_id}_history")
    if cache.get(f"{user_id}_actual_titles"):
        cache.delete(f"{user_id}_actual_titles")
    if cache.get(f"{user_id}_actual_urls"):
        cache.delete(f"{user_id}_actual_urls")
    if cache.get(f"{user_id}_token_limit"):
        cache.delete(f"{user_id}_token_limit")
    if cache.get(f"{user_id}_local_log_option"):
        cache.delete(f"{user_id}_local_log_option")
    if cache.get(f"{user_id}_cognito_user_id"):
        cache.delete(f"{user_id}_cognito_user_id")
    if cache.get(f"{user_id}_local_model"):
        cache.delete(f"{user_id}_local_model")
    if cache.get(f"{user_id}_local_history_maxlen"):
        cache.delete(f"{user_id}_local_history_maxlen")
    if cache.get(f"{user_id}_headers"):
        cache.delete(f"{user_id}_headers")

    if cache.get(f"{user_id}_closest_titles"):
        cache.delete(f"{user_id}_closest_titles")
        cache.delete(f"{user_id}_combined_scores")
        cache.delete(f"{user_id}_closest_vector_indices")

    if cache.get(f"{user_id}_matched_ids"):
        cache.delete(f"{user_id}_matched_ids")
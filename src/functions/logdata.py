import os
import csv
from src.schema.logging_config import logger

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def log_data(cognito_user_id, local_log_option, user_message, response, full_response_content, actual_urls, closest_titles, combined_scores, closest_vector_indices, matched_ids):
    logger.info(f"logdata has started with log_option: {local_log_option} and matched_ids: {matched_ids}")
    
    if local_log_option == 'fine_tune':
        try:
            # Check if the file is empty (i.e., if it's a new file)
            log_data_path = os.path.join(ROOT_DIR, f"dir_{cognito_user_id}", 'fine_tuning_data.csv')
            is_new_file = not os.path.exists(log_data_path) or os.stat(log_data_path).st_size == 0

            # Log the user messages and AI responses to a CSV file
            with open(log_data_path, 'a', newline='') as file:
                fieldnames = ['prompt', 'completion']
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                # Write the header only if the file is new
                if is_new_file:
                    writer.writeheader()

                if 'choices' in response:  # responseがstreamでない場合
                    ai_response_content = response['choices'][0]['message']['content']
                else:  # responseがstreamの場合
                    ai_response_content = full_response_content
     
                writer.writerow({
                    'prompt': user_message,
                    'completion': ai_response_content
                })
        except Exception as e:
            logger.error(f"Error while logging in 'fine_tune' option: {e}")

    elif local_log_option == 'vector_score_log' and not matched_ids:
        try:
            # Check if the file is empty (i.e., if it's a new file)
            log_data_path = os.path.join(ROOT_DIR, f"dir_{cognito_user_id}", 'vector_score.csv')
            is_new_file = not os.path.exists(log_data_path) or os.stat(log_data_path).st_size == 0

            # Log the distances and URLs etc... to a CSV file
            with open(log_data_path, 'a', newline='') as file:
                fieldnames = ['user_message', 'title', 'id', 'score', 'actual_referred_urls', 'ai_response']
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                # Write the header only if the file is new
                if is_new_file:
                    writer.writeheader()

                # Ensure all lists have the same length
                while len(actual_urls) < len(closest_titles):
                    actual_urls.append(None)

                adjusted_ids = [idx + 2 for idx in closest_vector_indices]
                for title, each_score, each_url, adj_id in zip(closest_titles, combined_scores, actual_urls, adjusted_ids):
                    
                    if 'choices' in response:  # responseがstreamでない場合
                        ai_response_content = response['choices'][0]['message']['content']
                    else:  # responseがstreamの場合
                        ai_response_content = full_response_content
                    writer.writerow({
                        'user_message': user_message, 
                        'title': title,
                        'id': adj_id,
                        'score': each_score,
                        'actual_referred_urls': each_url,
                        'ai_response': ai_response_content
                    })
                
                logger.info("Finished writing to file.")
        except Exception as e:
            logger.error(f"Error while logging in 'vector_score_log' option: {e}")
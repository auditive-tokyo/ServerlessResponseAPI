import os
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# folder_name = ""
# settings_path = ""

# Cognito IDから動的にフォルダー名を設定する
def generate_folder_name(cognito_user_id):
    return f"dir_{cognito_user_id}"

def update_settings_path(cognito_user_id):
    folder_name = generate_folder_name(cognito_user_id)
    settings_path = os.path.join(ROOT_DIR, folder_name, 'settings.json')
    return settings_path
    
# reference.jsonの読み込み
def get_file_path(cognito_user_id):
    folder_name = generate_folder_name(cognito_user_id)
    file_path = os.path.join(ROOT_DIR, folder_name, 'reference.json')
    with open(file_path, 'r') as f:
        data = json.load(f)
        documents = data
    return file_path, data, documents
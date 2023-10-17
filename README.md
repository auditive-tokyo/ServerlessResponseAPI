# DeepFaissChat

**English Below**

---

## 使用方法

- プロジェクトフォルダ内でPython 3の仮想環境を作成します。
- 仮想環境から `pip install -r requirements.txt` コマンドで必要なライブラリをインストールします。
- 仮想環境から `python app.py` または `nohup python app.py &` を実行してアプリケーションを立ち上げます。

### 注意: Server Sent Eventのサポートが必須です
このアプリケーションはServer Sent Eventを使用しているため、Server Sent Eventがサポートされている環境でのみ動作します。EC2などのインスタンスを使用する場合は、Apacheサーバーを設定し、ポート8080でリバースプロキシを設定してください。

### ローカルでのテスト
ローカルでテストする場合は、MAMPを使用してWPプラグインのクライアントを立ち上げれば、ローカル環境でもテストが可能です。

## 注意事項

このリポジトリは、私のコーディングスキルを評価していただく目的で特定のエンティティに共有しております。以下のポイントにご留意いただけますと幸いです。

### 使用について

- このコードは、コーディング能力の評価を目的としています。
- 評価以外での使用はご遠慮いただきますようお願い申し上げます。

### 秘密保持について

- このリポジトリを閲覧することで、秘密保持に関する同意をいただいたものとさせていただきます。
- 評価以外の目的で第三者との共有はご遠慮いただきますようお願い申し上げます。

---

## How to Use

- Create a Python 3 virtual environment within the project folder.
- From the virtual environment, run the `pip install -r requirements.txt` command to install the required libraries.
- From the virtual environment, execute `python app.py` or `nohup python app.py &` to launch the application.

### Note: Server Sent Event Support is Required
This application utilizes Server Sent Events and will only function in environments that support this technology. If you are using an instance like EC2, please configure an Apache server and set up a reverse proxy on port 8080.

### Local Testing
For local testing, you can launch the WP plugin client using MAMP, making it possible to test in a local environment.

## Important Notice

This repository is shared with specific entities for the purpose of evaluating my coding skills. Your attention to the following points would be greatly appreciated.

### Usage Guidelines

- This code is intended solely for the evaluation of coding skills.
- Please refrain from using it for any purpose other than evaluation.

### Confidentiality Agreement

- By accessing this repository, you agree to adhere to a confidentiality agreement.
- Please refrain from sharing this code with third parties for purposes other than evaluation.

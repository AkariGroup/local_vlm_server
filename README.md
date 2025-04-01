# Local VLM Server

ローカル環境で動作するVisual Language Model (VLM)サーバー。[SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)を使用して、画像理解と自然言語での応答を提供する。  

## インストール

```bash
pip install -r requirements.txt
```
## 使用方法

### サーバーの起動

```bash
python vlm_server.py [--model MODEL_NAME]
```

#### パラメータ

- `--model`: 使用するVLMモデル名（デフォルト: "HuggingFaceTB/SmolVLM-256M-Instruct"）  
    使用可能なモデルは下記。  
    - "HuggingFaceTB/SmolVLM-256M-Instruct",
    - "HuggingFaceTB/SmolVLM-500M-Instruct",
    - "vikhyatk/moondream2"

### クライアントの使用例

例)
```bash
python local_vlm_example.py --image_path image/akari.jpg --prompt "Please describe this image."
```

#### パラメータ

- `--image_path`: 画像ファイルのパス（複数指定可）
- `--prompt`: VLMへ送信するプロンプト（指定がない場合はデフォルトのプロンプトが使用されます。英語のみ対応）

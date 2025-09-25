# Local VLM Server

ローカル環境で動作するVisual Language Model (VLM)サーバー。[SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)や[moondream2](https://huggingface.co/vikhyatk/moondream2)を使用して、画像理解と自然言語での応答を提供する。  


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
    - [SmolVLM Instruct シリーズ](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)  
      `HuggingFaceTB/SmolVLM-256M-Instruct`, `HuggingFaceTB/SmolVLM-500M-Instruct`
    - [moondream2](https://huggingface.co/vikhyatk/moondream2)  
      `vikhyatk/moondream2`
    - [Heron NVILA Lite シリーズ](https://huggingface.co/turing-motors/Heron-NVILA-Lite-1B)  
      `turing-motors/Heron-NVILA-Lite-1B`, `turing-motors/Heron-NVILA-Lite-2B`, `turing-motors/Heron-NVILA-Lite-15B`
    - [FastVLM シリーズ](https://huggingface.co/apple/FastVLM-0.5B)  
      `apple/FastVLM-0.5B`, `apple/FastVLM-1.5B`, `apple/FastVLM-7B`
    - [Qwen2.5-VL Instruct シリーズ](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)  
      `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-32B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`
    - [LFM2-VL シリーズ](https://huggingface.co/LiquidAI/LFM2-VL-1.6B)  
      `LiquidAI/LFM2-VL-450M`, `LiquidAI/LFM2-VL-1.6B`
    - [InternVL3 シリーズ](https://huggingface.co/OpenGVLab/InternVL3-1B)  
      `OpenGVLab/InternVL3-1B`, `OpenGVLab/InternVL3-2B`, `OpenGVLab/InternVL3-8B`, `OpenGVLab/InternVL3-9B`, `OpenGVLab/InternVL3-14B`, `OpenGVLab/InternVL3-38B`, `OpenGVLab/InternVL3-78B`

### クライアントの使用例

例)
```bash
python local_vlm_example.py --image_path image/akari.jpg --prompt "Please describe this image."
```

#### パラメータ

- `--image_path`: 画像ファイルのパス（複数指定可）
- `--prompt`: VLMへ送信するプロンプト（指定がない場合はデフォルトのプロンプトが使用されます。英語のみ対応）

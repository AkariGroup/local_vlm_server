import argparse
import os
import sys
import time
import json
import base64
from datetime import datetime
from concurrent import futures
from typing import Any

import grpc
from lib.vlm import get_vlm_category

sys.path.append(os.path.join(os.path.dirname(__file__), "lib/grpc"))
import local_vlm_server_pb2
import local_vlm_server_pb2_grpc


class LocalVlmServer(local_vlm_server_pb2_grpc.LocalVlmServerServiceServicer):
    """
    VLMに画像を送信し、レスポンスを返すサーバ
    """

    def __init__(self, vlm: Any, save_dir: str = None) -> None:
        self.vlm = vlm
        self.save_dir = save_dir
        self.counter = 0
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def SendImage(
        self,
        request: local_vlm_server_pb2.SendImageRequest(),
        context: grpc.ServicerContext,
    ) -> local_vlm_server_pb2.SendImageReply:
        prompt = None
        if request.HasField("prompt"):
            prompt = request.prompt
        try:
            # Base64エンコードされた画像を処理
            images_list = list(request.images)  # RepeatedScalarContainerをリストに変換
            start_time = time.time()
            response = self.vlm.chat(images=images_list, prompt=prompt)
            interval = time.time() - start_time

            # 保存モードが有効な場合
            if self.save_dir:
                timestamp = datetime.now().isoformat()
                for idx, image_base64 in enumerate(images_list):
                    # 画像の保存
                    image_data = base64.b64decode(image_base64)
                    image_path = os.path.join(self.save_dir, f"{self.counter:04d}.jpg")
                    with open(image_path, "wb") as f:
                        f.write(image_data)

                    # JSON保存
                    json_data = {
                        "timestamp": timestamp,
                        "prompt": request.prompt,
                        "response": response,
                        "response_time": interval,
                        "image_file": f"{self.counter:04d}.jpg",
                    }
                    json_path = os.path.join(self.save_dir, f"{self.counter:04d}.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)

                    self.counter += 1

            return local_vlm_server_pb2.SendImageReply(response=response)
        except Exception as e:
            print(f"Error in SendImage: {str(e)}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Vlm model name",
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        help="Directory to save images and responses",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    vlm_category = get_vlm_category(args.model)
    if vlm_category == "smolvlm":
        from lib.smolvlm import SmolVlm as Vlm
    elif vlm_category == "moondream":
        from lib.moondream import Moondream as Vlm
    elif vlm_category == "heron":
        from lib.heron import HeronNvilaLite as Vlm
    elif vlm_category == "fastvlm":
        from lib.fastvlm import FastVlm as Vlm
    elif vlm_category == "qwen":
        from lib.qwen import Qwen as Vlm
    elif vlm_category == "lfm2vl":
        from lib.lfm2vl import Lfm2Vl as Vlm
    elif vlm_category == "internvl3":
        from lib.internvl3 import InrernVl3 as Vlm
    else:
        raise ValueError(f"Unknown VLM category: {vlm_category}")
    vlm = Vlm(args.model)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    local_vlm_server_pb2_grpc.add_LocalVlmServerServiceServicer_to_server(
        LocalVlmServer(vlm, args.save_dir), server
    )
    port = "10020"
    server.add_insecure_port("[::]:" + port)
    server.start()
    print(f"local_vlm_server start. port: {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()

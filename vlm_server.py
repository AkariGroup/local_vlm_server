import argparse
import os
import sys
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

    def __init__(self, vlm: Any) -> None:
        self.vlm = vlm

    def SendImage(
        self,
        request: local_vlm_server_pb2.SendImageRequest(),
        context: grpc.ServicerContext,
    ) -> local_vlm_server_pb2.SendImageReply:
        try:
            # Base64エンコードされた画像を処理
            images_list = list(request.images)  # RepeatedScalarContainerをリストに変換
            response = self.vlm.chat(
                images_list, request.prompt
            )
            return local_vlm_server_pb2.SendImageReply(response=response)
        except Exception as e:
            print(f"Error in SendImage: {str(e)}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Vlm model name",
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        type=str,
    )
    args = parser.parse_args()
    vlm_category = get_vlm_category(args.model)
    if vlm_category == "smolvlm":
        from lib.smolvlm import SmolVlm as Vlm
    elif vlm_category == "moondream":
        from lib.moondream import Moondream as Vlm
    else:
        raise ValueError(f"Unknown VLM category: {vlm_category}")
    vlm = Vlm(args.model)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    local_vlm_server_pb2_grpc.add_LocalVlmServerServiceServicer_to_server(
        LocalVlmServer(vlm), server
    )
    port = "10020"
    server.add_insecure_port("[::]:" + port)
    server.start()
    print(f"local_vlm_server start. port: {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()

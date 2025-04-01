import argparse
import base64
import grpc
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "lib/grpc"))
import local_vlm_server_pb2
import local_vlm_server_pb2_grpc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        help="List of paths to image files",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--prompt",
        help="Prompt to send to VLM",
        type=str,
    )
    args = parser.parse_args()

    # gRPCチャネルを作成
    channel = grpc.insecure_channel("localhost:10020")
    stub = local_vlm_server_pb2_grpc.LocalVlmServerServiceStub(channel)
    encoded_images = []

    # 画像をbase64エンコード
    for image_path in args.image_path:
        if not os.path.exists(image_path):
            print(f"Image file does not exist: {image_path}")
            continue
        try:
            with open(image_path, "rb") as image_file:
                # 画像データをbase64エンコード（プレフィックスなし）
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                encoded_images.append(encoded)
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            continue

    if not encoded_images:
        print("No valid images to process")
        return

    # 画像とプロンプトを送信
    request = local_vlm_server_pb2.SendImageRequest(
        images=encoded_images,
        prompt=args.prompt,
    )
    start_time = time.time()
    try:
        response = stub.SendImage(request)
        print(f"Response: {response.response}")
        print(f"Time taken: {time.time() - start_time:.2f}s")
    except grpc.RpcError as e:
        print(f"RPC error: {e}")


if __name__ == "__main__":
    main()

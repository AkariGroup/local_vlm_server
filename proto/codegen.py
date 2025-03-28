from grpc.tools import protoc

protoc.main(
    (
        "",
        "-I.",
        "--python_out=../lib/grpc",
        "--grpc_python_out=../lib/grpc",
        "local_vlm_server.proto",
    )
)

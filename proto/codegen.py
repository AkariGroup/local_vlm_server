from grpc.tools import protoc

protoc.main(
    (
        "",
        "-I.",
        "--python_out=../lib/grpc",
        "--grpc_python_out=../lib/grpc",
        "streamlit_server.proto",
    )
)

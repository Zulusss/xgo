TORCH_DIR="/home/roman/xgo/xgo/xai/libtorch"

export CGO_LDFLAGS="-L${TORCH_DIR}/lib -Wl,-rpath,${TORCH_DIR}/lib"
export CGO_CXXFLAGS="-I${TORCH_DIR}/include -I${TORCH_DIR}/include/torch/csrc/api/include"

go build .
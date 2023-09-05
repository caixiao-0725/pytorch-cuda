#include <torch/extension.h>
torch::Tensor trilinear_interpolation(
    torch::Tensor feats,
    torch::Tensor point) {
    return feats;
}

int add(int i, int j) {
    return i + j;
}
//TORCH_EXTENSION_NAME = cppcuda_tutorial  这个名称要和setup.py中的name一致
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trilinear_interpolation", &trilinear_interpolation);
    m.def("add", &add);
}



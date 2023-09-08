#include "include/utils.h"

torch::Tensor trilinear_interpolation_fw(
    const torch::Tensor feats,
    const torch::Tensor point
) {
    CHECK_INPUT(feats);
    CHECK_INPUT(point);
    return trilinear_fw_cu(feats,point);
}

torch::Tensor trilinear_interpolation_bw(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor point
) {
    CHECK_INPUT(feats);
    CHECK_INPUT(point);
    return trilinear_bw_cu(dL_dfeat_interp,feats,point);
}

int add(int i, int j) {
    return i + j;
}
//TORCH_EXTENSION_NAME = cppcuda_tutorial  这个名称要和setup.py中的name一致
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trilinear_interpolation_fw", &trilinear_interpolation_fw);
    m.def("trilinear_interpolation_bw", &trilinear_interpolation_bw);
    m.def("add", &add);
}



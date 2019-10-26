
#include <torch/extension.h>

torch::Tensor PoissonDisk(torch::Tensor vertex, torch::Tensor faces, torch::Tensor out, unsigned int sampleNum,  float rad);
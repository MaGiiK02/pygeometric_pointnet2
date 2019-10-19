
#include <torch/extension.h>

torch::Tensor PoissonDisk(torch::Tensor vertex, torch::Tensor faces, unsigned int sampleNum,  float rad);
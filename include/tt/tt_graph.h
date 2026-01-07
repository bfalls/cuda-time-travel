#ifndef TT_GRAPH_H
#define TT_GRAPH_H

#include <cuda_runtime.h>

namespace tt {

class GraphSession {
public:
    bool begin_capture(cudaStream_t stream);
    bool end_capture();
    bool launch(cudaStream_t stream);
    void destroy();

private:
    cudaStream_t capture_stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
};

} // namespace tt

#endif // TT_GRAPH_H

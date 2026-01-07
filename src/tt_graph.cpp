#include "tt/tt_graph.h"

namespace tt {

bool GraphSession::begin_capture(cudaStream_t stream) {
    destroy();
    capture_stream_ = stream;
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    return err == cudaSuccess;
}

bool GraphSession::end_capture() {
    cudaStream_t stream = capture_stream_;
    if (!stream) {
        return false;
    }
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        if (graph) {
            cudaGraphDestroy(graph);
        }
        return false;
    }

    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        return false;
    }

    destroy();
    graph_ = graph;
    graph_exec_ = exec;
    capture_stream_ = nullptr;
    return true;
}

bool GraphSession::launch(cudaStream_t stream) {
    if (!graph_exec_) {
        return false;
    }
    return cudaGraphLaunch(graph_exec_, stream) == cudaSuccess;
}

void GraphSession::destroy() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    capture_stream_ = nullptr;
}

} // namespace tt

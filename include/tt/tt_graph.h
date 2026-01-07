#ifndef TT_GRAPH_H
#define TT_GRAPH_H

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace tt {

struct GraphNodeInfo {
    uint32_t id = 0;
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    std::string name;
};

enum class GraphUpdateReason : uint32_t {
    kOk = 0,
    kNodeTypeUnsupported,
    kKernelParamLayoutChanged,
    kGraphUpdateNotSupported,
    kUpdateFailedRuntimeError
};

struct GraphUpdateStatus {
    bool applied = false;
    GraphUpdateReason reason = GraphUpdateReason::kOk;
    cudaError_t cuda_error = cudaSuccess;
};

void RegisterGraphKernelTag(const void* func, const char* name);
const char* GraphUpdateReasonString(GraphUpdateReason reason);

class GraphSession {
public:
    bool begin_capture(cudaStream_t stream);
    bool end_capture();
    bool launch(cudaStream_t stream);
    void destroy();
    const std::vector<GraphNodeInfo>& get_nodes() const { return node_infos_; }
    bool set_node_name(uint32_t node_id, const char* name);
    bool update_kernel_node_params(uint32_t node_id, const cudaKernelNodeParams& params, GraphUpdateStatus* out);
    bool update_memcpy_node_params(uint32_t node_id, const cudaMemcpy3DParms& params, GraphUpdateStatus* out);
    bool update_exec(cudaGraph_t new_graph, GraphUpdateStatus* out);
    bool get_kernel_node_params(uint32_t node_id, cudaKernelNodeParams* out) const;
    bool get_memcpy_node_params(uint32_t node_id, cudaMemcpy3DParms* out) const;
    cudaGraphExec_t exec() const { return graph_exec_; }
    cudaGraph_t graph() const { return graph_; }

private:
    void refresh_nodes();
    cudaGraphNode_t get_node_by_id(uint32_t node_id) const;
    void set_update_status(GraphUpdateStatus* out, bool applied, GraphUpdateReason reason, cudaError_t err) const;
    cudaStream_t capture_stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    std::vector<cudaGraphNode_t> nodes_{};
    std::vector<GraphNodeInfo> node_infos_{};
};

} // namespace tt

#endif // TT_GRAPH_H

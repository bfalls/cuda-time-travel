#include "tt/tt_graph.h"

#include <cstdio>

namespace tt {

namespace {

struct KernelTag {
    const void* func = nullptr;
    std::string name{};
};

std::vector<KernelTag>& kernel_tags() {
    static std::vector<KernelTag> tags;
    return tags;
}

const char* lookup_kernel_tag(const void* func) {
    if (!func) {
        return nullptr;
    }
    for (const auto& tag : kernel_tags()) {
        if (tag.func == func) {
            return tag.name.c_str();
        }
    }
    return nullptr;
}

const char* default_node_name(cudaGraphNodeType type) {
    switch (type) {
    case cudaGraphNodeTypeKernel:
        return "kernel";
    case cudaGraphNodeTypeMemcpy:
        return "memcpy";
    case cudaGraphNodeTypeMemset:
        return "memset";
    case cudaGraphNodeTypeHost:
        return "host";
    case cudaGraphNodeTypeGraph:
        return "graph";
    case cudaGraphNodeTypeEmpty:
        return "empty";
    default:
        return "node";
    }
}

} // namespace

void RegisterGraphKernelTag(const void* func, const char* name) {
    if (!func || !name) {
        return;
    }
    auto& tags = kernel_tags();
    for (auto& tag : tags) {
        if (tag.func == func) {
            tag.name = name;
            return;
        }
    }
    tags.push_back({func, name});
}

const char* GraphUpdateReasonString(GraphUpdateReason reason) {
    switch (reason) {
    case GraphUpdateReason::kOk:
        return "OK";
    case GraphUpdateReason::kNodeTypeUnsupported:
        return "NODE_TYPE_UNSUPPORTED";
    case GraphUpdateReason::kKernelParamLayoutChanged:
        return "KERNEL_PARAM_LAYOUT_CHANGED";
    case GraphUpdateReason::kGraphUpdateNotSupported:
        return "GRAPH_UPDATE_NOT_SUPPORTED";
    case GraphUpdateReason::kUpdateFailedRuntimeError:
        return "UPDATE_FAILED_RUNTIME_ERROR";
    default:
        return "UNKNOWN";
    }
}

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
    refresh_nodes();
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
    nodes_.clear();
    node_infos_.clear();
}

bool GraphSession::set_node_name(uint32_t node_id, const char* name) {
    if (!name) {
        return false;
    }
    if (node_id >= node_infos_.size()) {
        return false;
    }
    node_infos_[node_id].name = name;
    return true;
}

bool GraphSession::update_kernel_node_params(uint32_t node_id,
    const cudaKernelNodeParams& params,
    GraphUpdateStatus* out) {
    if (!graph_exec_) {
        set_update_status(out, false, GraphUpdateReason::kUpdateFailedRuntimeError, cudaErrorInvalidResourceHandle);
        return false;
    }
    cudaGraphNode_t node = get_node_by_id(node_id);
    if (!node) {
        set_update_status(out, false, GraphUpdateReason::kNodeTypeUnsupported, cudaErrorInvalidValue);
        return false;
    }
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess || type != cudaGraphNodeTypeKernel) {
        set_update_status(out, false, GraphUpdateReason::kNodeTypeUnsupported, cudaErrorInvalidValue);
        return false;
    }
    cudaError_t err = cudaGraphExecKernelNodeSetParams(graph_exec_, node, &params);
    if (err == cudaSuccess) {
        set_update_status(out, true, GraphUpdateReason::kOk, err);
        return true;
    }
    if (err == cudaErrorNotSupported) {
        set_update_status(out, false, GraphUpdateReason::kGraphUpdateNotSupported, err);
        return false;
    }
    if (err == cudaErrorInvalidValue) {
        set_update_status(out, false, GraphUpdateReason::kKernelParamLayoutChanged, err);
        return false;
    }
    set_update_status(out, false, GraphUpdateReason::kUpdateFailedRuntimeError, err);
    return false;
}

bool GraphSession::update_memcpy_node_params(uint32_t node_id,
    const cudaMemcpy3DParms& params,
    GraphUpdateStatus* out) {
    if (!graph_exec_) {
        set_update_status(out, false, GraphUpdateReason::kUpdateFailedRuntimeError, cudaErrorInvalidResourceHandle);
        return false;
    }
    cudaGraphNode_t node = get_node_by_id(node_id);
    if (!node) {
        set_update_status(out, false, GraphUpdateReason::kNodeTypeUnsupported, cudaErrorInvalidValue);
        return false;
    }
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess || type != cudaGraphNodeTypeMemcpy) {
        set_update_status(out, false, GraphUpdateReason::kNodeTypeUnsupported, cudaErrorInvalidValue);
        return false;
    }
    cudaError_t err = cudaGraphExecMemcpyNodeSetParams(graph_exec_, node, &params);
    if (err == cudaSuccess) {
        set_update_status(out, true, GraphUpdateReason::kOk, err);
        return true;
    }
    if (err == cudaErrorNotSupported) {
        set_update_status(out, false, GraphUpdateReason::kGraphUpdateNotSupported, err);
        return false;
    }
    if (err == cudaErrorInvalidValue) {
        set_update_status(out, false, GraphUpdateReason::kKernelParamLayoutChanged, err);
        return false;
    }
    set_update_status(out, false, GraphUpdateReason::kUpdateFailedRuntimeError, err);
    return false;
}

bool GraphSession::update_exec(cudaGraph_t new_graph, GraphUpdateStatus* out) {
    if (!graph_exec_ || !new_graph) {
        set_update_status(out, false, GraphUpdateReason::kUpdateFailedRuntimeError, cudaErrorInvalidResourceHandle);
        return false;
    }
    cudaGraphExecUpdateResultInfo result_info{};
    result_info.result = cudaGraphExecUpdateError;
    cudaError_t err = cudaGraphExecUpdate(graph_exec_, new_graph, &result_info);
    if (err == cudaSuccess && result_info.result == cudaGraphExecUpdateSuccess) {
        set_update_status(out, true, GraphUpdateReason::kOk, err);
        refresh_nodes();
        return true;
    }
    if (err == cudaErrorNotSupported) {
        set_update_status(out, false, GraphUpdateReason::kGraphUpdateNotSupported, err);
        return false;
    }
    if (err == cudaSuccess && result_info.result != cudaGraphExecUpdateSuccess) {
        set_update_status(out, false, GraphUpdateReason::kKernelParamLayoutChanged, err);
        return false;
    }
    set_update_status(out, false, GraphUpdateReason::kUpdateFailedRuntimeError, err);
    return false;
}

bool GraphSession::get_kernel_node_params(uint32_t node_id, cudaKernelNodeParams* out) const {
    if (!out || !graph_) {
        return false;
    }
    cudaGraphNode_t node = get_node_by_id(node_id);
    if (!node) {
        return false;
    }
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess || type != cudaGraphNodeTypeKernel) {
        return false;
    }
    return cudaGraphKernelNodeGetParams(node, out) == cudaSuccess;
}

bool GraphSession::get_memcpy_node_params(uint32_t node_id, cudaMemcpy3DParms* out) const {
    if (!out || !graph_) {
        return false;
    }
    cudaGraphNode_t node = get_node_by_id(node_id);
    if (!node) {
        return false;
    }
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess || type != cudaGraphNodeTypeMemcpy) {
        return false;
    }
    return cudaGraphMemcpyNodeGetParams(node, out) == cudaSuccess;
}

void GraphSession::refresh_nodes() {
    nodes_.clear();
    node_infos_.clear();
    if (!graph_) {
        return;
    }
    size_t node_count = 0;
    if (cudaGraphGetNodes(graph_, nullptr, &node_count) != cudaSuccess || node_count == 0) {
        return;
    }
    nodes_.resize(node_count);
    if (cudaGraphGetNodes(graph_, nodes_.data(), &node_count) != cudaSuccess) {
        nodes_.clear();
        return;
    }
    node_infos_.reserve(node_count);
    for (size_t i = 0; i < node_count; ++i) {
        cudaGraphNode_t node = nodes_[i];
        cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
        cudaGraphNodeGetType(node, &type);
        std::string name = default_node_name(type);
        if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params{};
            if (cudaGraphKernelNodeGetParams(node, &params) == cudaSuccess) {
                if (const char* tag = lookup_kernel_tag(params.func)) {
                    name = tag;
                }
            }
        }
        GraphNodeInfo info{};
        info.id = static_cast<uint32_t>(i);
        info.type = type;
        info.name = name;
        node_infos_.push_back(std::move(info));
    }
}

cudaGraphNode_t GraphSession::get_node_by_id(uint32_t node_id) const {
    if (node_id >= nodes_.size()) {
        return nullptr;
    }
    return nodes_[node_id];
}

void GraphSession::set_update_status(GraphUpdateStatus* out,
    bool applied,
    GraphUpdateReason reason,
    cudaError_t err) const {
    if (!out) {
        return;
    }
    out->applied = applied;
    out->reason = reason;
    out->cuda_error = err;
}

} // namespace tt

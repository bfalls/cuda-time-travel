#include "tt/tt_trace.h"
#include "tt/ttrecorder.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>

namespace {

	bool check_cuda(cudaError_t err, const char* label) {
		if (err == cudaSuccess) {
			return true;
		}
		std::printf("CUDA error at %s: %s\n", label, cudaGetErrorString(err));
		return false;
	}

	bool has_flag(const char* flag) {
		for (int i = 1; i < __argc; ++i) {
			if (std::strcmp(__argv[i], flag) == 0) return true;
		}
		return false;
	}

	// Supports: --demo=all, --demo=1, --demo=2, --demo=3, --demo=1,2,3 (commas)
	// Default: all (if flag is absent)
	uint32_t parse_demo_mask() {
		const char* prefix = "--demo=";
		const size_t prefix_len = std::strlen(prefix);

		const char* value = nullptr;
		for (int i = 1; i < __argc; ++i) {
			const char* arg = __argv[i];
			if (std::strncmp(arg, prefix, prefix_len) == 0) {
				value = arg + prefix_len;
				break;
			}
		}
		if (!value || value[0] == '\0') {
			// Default: run all demos
			return (1u << 1) | (1u << 2) | (1u << 3);
		}

		if (std::strcmp(value, "all") == 0) {
			return (1u << 1) | (1u << 2) | (1u << 3);
		}

		uint32_t mask = 0;
		// Parse comma-separated digits: 1,2,3
		const char* p = value;
		while (*p) {
			while (*p == ' ' || *p == '\t' || *p == ',') ++p;
			if (*p == '\0') break;

			if (*p == '1') { mask |= (1u << 1); ++p; continue; }
			if (*p == '2') { mask |= (1u << 2); ++p; continue; }
			if (*p == '3') { mask |= (1u << 3); ++p; continue; }

			// Unknown token: ignore the rest (keep it simple and deterministic)
			break;
		}

		// If parsing yielded nothing, fall back to all.
		if (mask == 0) {
			mask = (1u << 1) | (1u << 2) | (1u << 3);
		}
		return mask;
	}

	const char* parse_tt_show_list() {
		const char* prefix = "--tt-show=";
		const size_t prefix_len = std::strlen(prefix);
		for (int i = 1; i < __argc; ++i) {
			const char* arg = __argv[i];
			if (std::strncmp(arg, prefix, prefix_len) == 0) {
				return arg + prefix_len;
			}
		}
		return nullptr;
	}

	void append_unique_checkpoint(std::vector<uint32_t>& out, uint32_t checkpoint) {
		for (uint32_t existing : out) {
			if (existing == checkpoint) {
				return;
			}
		}
		out.push_back(checkpoint);
	}

	void resolve_tt_show_checkpoints(const char* list,
		uint32_t first_bad,
		uint32_t checkpoint_count,
		std::vector<uint32_t>& out) {
		out.clear();
		if (!list || list[0] == '\0') {
			return;
		}

		const char* p = list;
		while (*p) {
			while (*p == ' ' || *p == '\t' || *p == ',') ++p;
			if (*p == '\0') break;

			const char* token_start = p;
			while (*p && *p != ',') ++p;
			const char* token_end = p;

			while (token_end > token_start && (token_end[-1] == ' ' || token_end[-1] == '\t')) {
				--token_end;
			}
			const size_t token_len = static_cast<size_t>(token_end - token_start);
			if (token_len == 0) {
				continue;
			}

			bool has_value = false;
			uint32_t value = 0;

			if (token_len == 8 && std::strncmp(token_start, "firstbad", 8) == 0) {
				value = first_bad;
				has_value = true;
			}
			else if (token_len == 4 && std::strncmp(token_start, "prev", 4) == 0) {
				if (first_bad > 0) {
					value = first_bad - 1u;
					has_value = true;
				}
			}
			else {
				uint64_t parsed = 0;
				bool valid = true;
				bool any = false;
				for (const char* c = token_start; c < token_end; ++c) {
					if (*c < '0' || *c > '9') {
						valid = false;
						break;
					}
					any = true;
					parsed = parsed * 10u + static_cast<uint64_t>(*c - '0');
					if (parsed > 0xFFFFFFFFull) {
						valid = false;
						break;
					}
				}
				if (valid && any) {
					value = static_cast<uint32_t>(parsed);
					has_value = true;
				}
			}

			if (has_value && value < checkpoint_count) {
				append_unique_checkpoint(out, value);
			}
		}
	}

	const char* recorder_status_to_string(tt::RecorderStatus status) {
		switch (status) {
		case tt::RecorderStatus::kOk: return "ok";
		case tt::RecorderStatus::kNotInitialized: return "not initialized";
		case tt::RecorderStatus::kInvalidConfig: return "invalid config";
		case tt::RecorderStatus::kInvalidDependency: return "invalid dependency";
		case tt::RecorderStatus::kCudaError: return "cuda error";
		case tt::RecorderStatus::kBackpressure: return "backpressure";
		case tt::RecorderStatus::kRingTooSmall: return "ring too small";
		case tt::RecorderStatus::kRingCorrupt: return "ring corrupt";
		case tt::RecorderStatus::kInvalidHeader: return "invalid header";
		case tt::RecorderStatus::kInvalidChunkType: return "invalid chunk type";
		case tt::RecorderStatus::kInvalidPayload: return "invalid payload";
		case tt::RecorderStatus::kAlignmentError: return "alignment error";
		case tt::RecorderStatus::kCheckpointNotFound: return "checkpoint not found";
		case tt::RecorderStatus::kCheckpointDropped: return "checkpoint dropped";
		case tt::RecorderStatus::kDeterminismViolation: return "determinism violation";
		default: return "unknown";
		}
	}

	__global__ void write_pattern_kernel(uint32_t* data, uint32_t count, uint32_t checkpoint) {
		const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < count) {
			data[idx] = checkpoint ^ (idx * 2654435761u);
		}
	}

	bool verify_pattern(const std::vector<uint32_t>& data, uint32_t checkpoint) {
		for (uint32_t i = 0; i < data.size(); ++i) {
			const uint32_t expected = checkpoint ^ (i * 2654435761u);
			if (data[i] != expected) {
				return false;
			}
		}
		return true;
	}

	void add_event(tt::TraceCollector& trace,
		const char* name,
		const char* cat,
		double ts_us,
		double dur_us,
		uint32_t checkpoint,
		uint32_t tid) {
		tt::TraceEvent event{};
		event.name = name;
		event.cat = cat;
		event.ts_us = ts_us;
		event.dur_us = dur_us;
		event.pid = 1;
		event.tid = tid;
		event.args.push_back({ "checkpoint_id", std::to_string(checkpoint), false });
		trace.add_event(event);
	}

	bool run_checkpoint(uint32_t checkpoint,
		tt::TraceCollector& trace,
		tt::Recorder& recorder,
		cudaStream_t stream,
		uint32_t* d_buffer,
		uint32_t element_count,
		std::chrono::steady_clock::time_point trace_start) {

		const uint32_t threads = 256;
		const uint32_t blocks = (element_count + threads - 1u) / threads;

		const auto now_us = [&]() {
			return std::chrono::duration<double, std::micro>(
				std::chrono::steady_clock::now() - trace_start).count();
			};

		const double t0 = now_us();

		write_pattern_kernel << <blocks, threads, 0, stream >> > (d_buffer, element_count, checkpoint);
		if (!check_cuda(cudaGetLastError(), "launch write_pattern_kernel")) {
			return false;
		}
		add_event(trace, "write_kernel", "checkpoint", now_us(), 0.0, checkpoint, 0);

		if (!recorder.capture_checkpoint(stream)) {
			std::printf("capture_checkpoint failed at %u\n", checkpoint);
			return false;
		}
		add_event(trace, "capture_checkpoint", "checkpoint", now_us(), 0.0, checkpoint, 1);

		add_event(trace, "checkpoint", "checkpoint", t0, (now_us() - t0), checkpoint, 2);
		return true;
	}

	// Demo 2: small, realistic "why is this so slow?"
	// A debug-only sync accidentally runs every iteration due to '&' vs '&&'.
	bool run_sync_demo(bool sync_bug,
		tt::TraceCollector& trace,
		cudaStream_t stream,
		std::chrono::steady_clock::time_point trace_start) {

		const auto now_us = [&]() {
			return std::chrono::duration<double, std::micro>(
				std::chrono::steady_clock::now() - trace_start).count();
			};

		// Make kernels take a noticeable amount of time so an accidental sync is obvious.
		// (Realistic: "work is heavy enough that a sync fence hurts.")
		const uint32_t big_elements = 4u * 1024u * 1024u; // 4M uint32_t = 16MB
		const uint32_t big_bytes = big_elements * sizeof(uint32_t);

		uint32_t* d_big = nullptr;
		if (!check_cuda(cudaMalloc(&d_big, big_bytes), "cudaMalloc d_big")) {
			return false;
		}

		const uint32_t threads = 256;
		const uint32_t blocks = (big_elements + threads - 1u) / threads;

		// Keep debug_mode OFF in both runs. The only difference is '&' vs '&&'.
		const bool debug_mode = false;

		// Keep the demo short but visible.
		const uint32_t iters = 12;

		for (uint32_t i = 0; i < iters; ++i) {
			const double t_checkpoint0 = now_us();

			write_pattern_kernel << <blocks, threads, 0, stream >> > (d_big, big_elements, i);
			if (!check_cuda(cudaGetLastError(), "launch write_pattern_kernel (sync demo)")) {
				cudaFree(d_big);
				return false;
			}
			add_event(trace, "launch_kernel", "sync_demo", now_us(), 0.0, i, 0);

			// Measure how much time we spend in this "debug fence" decision.
			// Good: fence never runs (debug_mode=false) => tiny duration.
			// Bug: '&' forces evaluation of RHS => cudaStreamSynchronize runs every iter => big duration.
			const double t_fence0 = now_us();

			if (sync_bug) {
				// BUG: '&' does NOT short-circuit; RHS always executes.
				if (debug_mode & (cudaSuccess == cudaStreamSynchronize(stream))) {
					// no-op
				}
			}
			else {
				// Correct: '&&' short-circuits; RHS never executes when debug_mode=false.
				if (debug_mode && (cudaSuccess == cudaStreamSynchronize(stream))) {
					// no-op
				}
			}

			const double t_fence1 = now_us();
			add_event(trace, "debug_sync_fence", "sync_demo", t_fence0, (t_fence1 - t_fence0), i, 1);

			const double t_checkpoint1 = now_us();
			add_event(trace, "iter", "sync_demo", t_checkpoint0, (t_checkpoint1 - t_checkpoint0), i, 2);
		}

		if (!check_cuda(cudaStreamSynchronize(stream), "final sync (sync demo)")) {
			cudaFree(d_big);
			return false;
		}

		cudaFree(d_big);
		return true;
	}

	// ------------------------------------------------------------------------
	// Demo 3: deterministic "time-travel to find first bad checkpoint" bug.
	//
	// Honest mistake: storing an accumulating offset/counter in uint32_t when it
	// should be 64-bit. Works fine for a while, then silently wraps and remains
	// wrong thereafter. This is monotonic (good -> bad forever) without any
	// "if (checkpoint == K)" contrivance.
	// ------------------------------------------------------------------------

	__global__ void accumulate_offset_u32_kernel(uint32_t* data, uint32_t count, uint32_t add_per_checkpoint) {
		const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < count) {
			data[idx] += (add_per_checkpoint + (idx & 0xFFu));
		}
	}

	uint32_t find_first_bad_checkpoint_accum(tt::Recorder& recorder,
		cudaStream_t stream,
		uint32_t* d_buffer,
		uint32_t size_bytes,
		uint32_t checkpoint_count,
		uint32_t add_per_checkpoint,
		uint32_t element_count);

	uint32_t find_first_bad_checkpoint_accum_strict(tt::Recorder& recorder,
		cudaStream_t stream,
		uint32_t* d_buffer,
		uint32_t size_bytes,
		uint32_t checkpoint_count,
		uint32_t add_per_checkpoint,
		uint32_t element_count);

	void build_expected_accum_u64(std::vector<uint64_t>& host_ref_u64,
		uint32_t checkpoint,
		uint32_t add_per_checkpoint);

	bool find_first_mismatch_accum(const std::vector<uint32_t>& host_out,
		const std::vector<uint64_t>& host_ref_u64,
		uint32_t* out_index,
		uint64_t* out_expected,
		uint32_t* out_actual);

	bool find_first_mismatch_accum_u64(const std::vector<uint32_t>& host_out,
		const std::vector<uint64_t>& host_ref_u64,
		uint32_t* out_index,
		uint64_t* out_expected,
		uint32_t* out_actual);

	bool accum_checkpoint_is_good_strict(tt::Recorder& recorder,
		cudaStream_t stream,
		uint32_t* d_buffer,
		uint32_t size_bytes,
		uint32_t checkpoint,
		uint32_t add_per_checkpoint,
		std::vector<uint32_t>& host_out,
		std::vector<uint64_t>& host_ref_u64);

	static bool run_accum_overflow_demo(
		bool tt_debug,
		const char* tt_show_list,
		cudaStream_t stream,
		uint32_t* d_buffer,
		uint32_t element_count,
		uint32_t size_bytes,
		uint32_t checkpoint_count,
		uint32_t add_per_checkpoint,
		std::chrono::steady_clock::time_point trace_start,
		uint32_t* out_verified_checkpoints
	) {

		const auto now_us = [&]() {
			return std::chrono::duration<double, std::micro>(
				std::chrono::steady_clock::now() - trace_start).count();
			};

		// Recorder + trace are consolidated here (optional debug path).
		tt::Recorder recorder_local;
		tt::TraceCollector trace_local;

		const uint32_t per_chunk_bytes =
			(static_cast<uint32_t>(sizeof(tt::ChunkHeader)) + size_bytes + 31u) & ~31u;
		const uint32_t ring_bytes = per_chunk_bytes * checkpoint_count + 4096u;

		tt::RecorderConfig cfg{};
		cfg.ring_bytes = ring_bytes;
		cfg.checkpoint_capacity = checkpoint_count + 4u;
		cfg.region_capacity = 1u;
		cfg.retention_checkpoints = 0u;
		cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;

		if (!recorder_local.init(cfg)) {
			std::printf("[accum_demo] recorder init failed\n");
			trace_local.write("trace/tt_demo_accum_overflow-bug.json");
			return false;

		}
		if (!recorder_local.register_region(0, d_buffer, size_bytes, 1)) {
			std::printf("[accum_demo] register_region failed\n");
			recorder_local.shutdown();
			trace_local.write("trace/tt_demo_accum_overflow-bug.json");
			return false;

		}

		tt::Recorder* recorder = tt_debug ? &recorder_local : nullptr;
		tt::TraceCollector* trace = tt_debug ? &trace_local : nullptr;
		const bool enable_tt_show = tt_debug && tt_show_list && tt_show_list[0] != '\0';

		const double t_init0 = now_us();
		if (!check_cuda(cudaMemsetAsync(d_buffer, 0, size_bytes, stream), "cudaMemsetAsync (accum demo)")) {
			recorder_local.shutdown();
			trace_local.write("trace/tt_demo_accum_overflow-bug.json");
			return false;
		}
		if (!check_cuda(cudaStreamSynchronize(stream), "sync init (accum demo)")) {
			recorder_local.shutdown();
			trace_local.write("trace/tt_demo_accum_overflow-bug.json");
			return false;
		}
		if (tt_debug)
			add_event(*trace, "init_zero", "accum_demo", t_init0, (now_us() - t_init0), 0, 0);

		const uint32_t threads = 256;
		const uint32_t blocks = (element_count + threads - 1u) / threads;

		uint32_t verified_checkpoints = 0;
		std::vector<uint32_t> host_out(element_count, 0u);
		std::vector<uint64_t> host_ref(element_count, 0ull);
		bool found_bad = false;
		uint32_t last_checkpoint = 0;

		for (uint32_t checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
			const double t0 = now_us();
			last_checkpoint = checkpoint;

			accumulate_offset_u32_kernel << <blocks, threads, 0, stream >> > (d_buffer, element_count, add_per_checkpoint);
			if (!check_cuda(cudaGetLastError(), "launch accumulate_offset_u32_kernel")) {
				recorder_local.shutdown();
				trace_local.write("trace/tt_demo_accum_overflow-bug.json");
				return false;
			}
			if (tt_debug) {
				add_event(*trace, "accum_kernel", "accum_demo", now_us(), 0.0, checkpoint, 0);

				if (!(*recorder).capture_checkpoint(stream)) {
					std::printf("capture_checkpoint failed at %u (accum demo)\n", checkpoint);
					recorder_local.shutdown();
					trace_local.write("trace/tt_demo_accum_overflow-bug.json");
					return false;
				}
				add_event(*trace, "capture_checkpoint", "accum_demo", now_us(), 0.0, checkpoint, 1);
				add_event(*trace, "checkpoint", "accum_demo", t0, (now_us() - t0), checkpoint, 2);
			}

			if (!found_bad) {
				for (uint32_t i = 0; i < element_count; ++i) {
					host_ref[i] += static_cast<uint64_t>(add_per_checkpoint + (i & 0xFFu));
				}

				if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes,
					cudaMemcpyDeviceToHost), "memcpy out (accum demo verify)")) {
					break;
				}

				bool ok = true;
				for (uint32_t i = 0; i < element_count; ++i) {
					if (static_cast<uint64_t>(host_out[i]) != host_ref[i]) {
						ok = false;
						break;
					}
				}
				if (!ok) {
					found_bad = true;
					if (!enable_tt_show) {
						break;
					}
				}
			}

			if (!found_bad) {
				++verified_checkpoints;
			}
		}

		if (!check_cuda(cudaStreamSynchronize(stream), "final sync (accum demo)")) {
			recorder_local.shutdown();
			trace_local.write("trace/tt_demo_accum_overflow-bug.json");
			return false;
		}

		std::printf("verified_good_checkpoints=%u (checkpoints [0..%u] passed, expected %u total)\n",
			verified_checkpoints,
			verified_checkpoints == 0 ? 0u : (verified_checkpoints - 1u),
			checkpoint_count);

		if (out_verified_checkpoints) {
			*out_verified_checkpoints = verified_checkpoints;
		}

		const bool ok = (verified_checkpoints == checkpoint_count);
		trace_local.write("trace/tt_demo_accum_overflow-bug.json");

		if (!ok && tt_debug && verified_checkpoints < checkpoint_count) {
			const uint32_t first_bad = enable_tt_show
				? find_first_bad_checkpoint_accum_strict(recorder_local, stream, d_buffer,
					size_bytes, checkpoint_count, add_per_checkpoint, element_count)
				: find_first_bad_checkpoint_accum(recorder_local, stream, d_buffer,
					size_bytes, checkpoint_count, add_per_checkpoint, element_count);
			trace_local.write("trace/tt_demo_accum_overflow-ttdebug.json");
			std::printf("first_bad_checkpoint_index=%u\n", first_bad);

			if (enable_tt_show) {
				std::vector<uint32_t> show_checkpoints;
				resolve_tt_show_checkpoints(tt_show_list, first_bad, checkpoint_count, show_checkpoints);
				if (!show_checkpoints.empty()) {
					uint32_t max_checkpoint = last_checkpoint;
					for (uint32_t checkpoint : show_checkpoints) {
						if (checkpoint > max_checkpoint) {
							max_checkpoint = checkpoint;
						}
					}
					for (uint32_t checkpoint = last_checkpoint + 1u;
						checkpoint <= max_checkpoint && checkpoint < checkpoint_count;
						++checkpoint) {
						accumulate_offset_u32_kernel << <blocks, threads, 0, stream >> > (d_buffer, element_count, add_per_checkpoint);
						if (!check_cuda(cudaGetLastError(), "launch accumulate_offset_u32_kernel (tt show)")) {
							recorder_local.shutdown();
							trace_local.write("trace/tt_demo_accum_overflow-ttdebug.json");
							return false;
						}
						if (!recorder_local.capture_checkpoint(stream)) {
							std::printf("capture_checkpoint failed at %u (accum demo tt show)\n", checkpoint);
							recorder_local.shutdown();
							trace_local.write("trace/tt_demo_accum_overflow-ttdebug.json");
							return false;
						}
					}
					if (!check_cuda(cudaStreamSynchronize(stream), "sync (accum demo tt show)")) {
						recorder_local.shutdown();
						trace_local.write("trace/tt_demo_accum_overflow-ttdebug.json");
						return false;
					}
				}
				for (uint32_t checkpoint : show_checkpoints) {
					if (!recorder_local.rewind_to_checkpoint(checkpoint, stream)) {
						std::printf("[tt] inspect checkpoint %u: FAIL\n", checkpoint);
						std::printf("rewind failed: %s\n", recorder_status_to_string(recorder_local.last_status()));
						continue;
					}
					if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes,
						cudaMemcpyDeviceToHost), "memcpy out (accum demo inspect)")) {
						std::printf("[tt] inspect checkpoint %u: FAIL\n", checkpoint);
						std::printf("copy failed\n");
						continue;
					}

					build_expected_accum_u64(host_ref, checkpoint, add_per_checkpoint);
					uint32_t mismatch_index = 0;
					uint64_t expected = 0;
					uint32_t actual = 0;
					const bool mismatch = find_first_mismatch_accum_u64(host_out, host_ref,
						&mismatch_index, &expected, &actual);
					if (!mismatch) {
						std::printf("[tt] inspect checkpoint %u: OK\n", checkpoint);
					}
					else {
						std::printf("[tt] inspect checkpoint %u: FAIL\n", checkpoint);
						std::printf("first mismatch at element %u\n", mismatch_index);
						std::printf("expected=0x%016llx actual=0x%016llx\n",
							static_cast<unsigned long long>(expected),
							static_cast<unsigned long long>(actual));
					}
				}
			}
		}

		recorder_local.shutdown();
		return ok;
	}

	void build_expected_accum_u64(std::vector<uint64_t>& host_ref_u64,
		uint32_t checkpoint,
		uint32_t add_per_checkpoint) {
		const uint64_t mul = static_cast<uint64_t>(checkpoint) + 1ull;
		for (uint32_t i = 0; i < host_ref_u64.size(); ++i) {
			host_ref_u64[i] = mul * static_cast<uint64_t>(add_per_checkpoint + (i & 0xFFu));
		}
	}

	bool find_first_mismatch_accum(const std::vector<uint32_t>& host_out,
		const std::vector<uint64_t>& host_ref_u64,
		uint32_t* out_index,
		uint64_t* out_expected,
		uint32_t* out_actual) {
		for (uint32_t i = 0; i < host_out.size(); ++i) {
			const uint64_t expected = host_ref_u64[i];
			const uint32_t actual = host_out[i];
			if (actual != static_cast<uint32_t>(expected)) {
				if (out_index) *out_index = i;
				if (out_expected) *out_expected = expected;
				if (out_actual) *out_actual = actual;
				return true;
			}
		}
		return false;
	}

	bool find_first_mismatch_accum_u64(const std::vector<uint32_t>& host_out,
		const std::vector<uint64_t>& host_ref_u64,
		uint32_t* out_index,
		uint64_t* out_expected,
		uint32_t* out_actual) {
		for (uint32_t i = 0; i < host_out.size(); ++i) {
			const uint64_t expected = host_ref_u64[i];
			const uint64_t actual = static_cast<uint64_t>(host_out[i]);
			if (actual != expected) {
				if (out_index) *out_index = i;
				if (out_expected) *out_expected = expected;
				if (out_actual) *out_actual = static_cast<uint32_t>(actual);
				return true;
			}
		}
		return false;
	}

	bool accum_checkpoint_is_good(tt::Recorder& recorder,
		cudaStream_t stream,
		uint32_t* d_buffer,
		uint32_t size_bytes,
		uint32_t checkpoint,
		uint32_t add_per_checkpoint,
		std::vector<uint32_t>& host_out,
		std::vector<uint64_t>& host_ref_u64) {

		if (!recorder.rewind_to_checkpoint(checkpoint, stream)) {
			return false;
		}
		if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes,
			cudaMemcpyDeviceToHost), "memcpy out (accum demo rewind)")) {
			return false;
		}

		build_expected_accum_u64(host_ref_u64, checkpoint, add_per_checkpoint);
		return !find_first_mismatch_accum(host_out, host_ref_u64, nullptr, nullptr, nullptr);
	}

	bool accum_checkpoint_is_good_strict(tt::Recorder& recorder,
		cudaStream_t stream,
		uint32_t* d_buffer,
		uint32_t size_bytes,
		uint32_t checkpoint,
		uint32_t add_per_checkpoint,
		std::vector<uint32_t>& host_out,
		std::vector<uint64_t>& host_ref_u64) {

		if (!recorder.rewind_to_checkpoint(checkpoint, stream)) {
			return false;
		}
		if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes,
			cudaMemcpyDeviceToHost), "memcpy out (accum demo rewind strict)")) {
			return false;
		}

		build_expected_accum_u64(host_ref_u64, checkpoint, add_per_checkpoint);
		return !find_first_mismatch_accum_u64(host_out, host_ref_u64, nullptr, nullptr, nullptr);
	}

	uint32_t find_first_bad_checkpoint_accum(tt::Recorder& recorder,
		cudaStream_t stream,
		uint32_t* d_buffer,
		uint32_t size_bytes,
		uint32_t checkpoint_count,
		uint32_t add_per_checkpoint,
		uint32_t element_count) {

		std::vector<uint32_t> host_out(element_count, 0u);
		std::vector<uint64_t> host_ref_u64(element_count, 0ull);

		uint32_t lo = 0;
		uint32_t hi = checkpoint_count;
		while (lo < hi) {
			const uint32_t mid = lo + (hi - lo) / 2u;
			if (accum_checkpoint_is_good(recorder, stream, d_buffer, size_bytes, mid,
				add_per_checkpoint, host_out, host_ref_u64)) {
				lo = mid + 1u;
			}
			else {
				hi = mid;
			}
		}

		uint32_t first_bad = lo;
		const uint32_t linear_start = (first_bad > 2u) ? (first_bad - 2u) : 0u;
		for (uint32_t e = linear_start; e <= first_bad && e < checkpoint_count; ++e) {
			if (!accum_checkpoint_is_good(recorder, stream, d_buffer, size_bytes, e,
				add_per_checkpoint, host_out, host_ref_u64)) {
				first_bad = e;
				break;
			}
		}

		return first_bad;
	}

	uint32_t find_first_bad_checkpoint_accum_strict(tt::Recorder& recorder,
		cudaStream_t stream,
		uint32_t* d_buffer,
		uint32_t size_bytes,
		uint32_t checkpoint_count,
		uint32_t add_per_checkpoint,
		uint32_t element_count) {

		std::vector<uint32_t> host_out(element_count, 0u);
		std::vector<uint64_t> host_ref_u64(element_count, 0ull);

		uint32_t lo = 0;
		uint32_t hi = checkpoint_count;
		while (lo < hi) {
			const uint32_t mid = lo + (hi - lo) / 2u;
			if (accum_checkpoint_is_good_strict(recorder, stream, d_buffer, size_bytes, mid,
				add_per_checkpoint, host_out, host_ref_u64)) {
				lo = mid + 1u;
			}
			else {
				hi = mid;
			}
		}

		uint32_t first_bad = lo;
		const uint32_t linear_start = (first_bad > 2u) ? (first_bad - 2u) : 0u;
		for (uint32_t e = linear_start; e <= first_bad && e < checkpoint_count; ++e) {
			if (!accum_checkpoint_is_good_strict(recorder, stream, d_buffer, size_bytes, e,
				add_per_checkpoint, host_out, host_ref_u64)) {
				first_bad = e;
				break;
			}
		}

		return first_bad;
	}

} // namespace

int main() {
	const bool bug_mode = has_flag("--bug");           // off-by-one demo bug
	const bool sync_bug = has_flag("--sync-bug");      // sync fence demo bug
	const bool tt_debug = has_flag("--tt-debug");      // demo 3 postmortem debug
	const char* tt_show_list = parse_tt_show_list();

	const uint32_t demo_mask = parse_demo_mask();
	const bool run_demo1 = (demo_mask & (1u << 1)) != 0;
	const bool run_demo2 = (demo_mask & (1u << 2)) != 0;
	const bool run_demo3 = (demo_mask & (1u << 3)) != 0;

	const uint32_t element_count = 512;
	const uint32_t size_bytes = element_count * sizeof(uint32_t);
	const uint32_t checkpoint_count = 6;

	uint32_t* d_buffer = nullptr;
	if (!check_cuda(cudaMalloc(&d_buffer, size_bytes), "cudaMalloc buffer")) {
		return 1;
	}

	cudaStream_t stream = nullptr;
	if (!check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate")) {
		cudaFree(d_buffer);
		return 1;
	}

	const uint32_t per_chunk_bytes =
		(static_cast<uint32_t>(sizeof(tt::ChunkHeader)) + size_bytes + 31u) & ~31u;
	const uint32_t ring_bytes = per_chunk_bytes * checkpoint_count + 4096u;

	tt::Recorder recorder;
	tt::RecorderConfig cfg{};
	cfg.ring_bytes = ring_bytes;
	cfg.checkpoint_capacity = checkpoint_count + 2u;
	cfg.region_capacity = 1u;
	cfg.retention_checkpoints = 0u;
	cfg.overwrite_mode = tt::OverwriteMode::kDropOldest;

	if (!recorder.init(cfg)) {
		std::printf("recorder init failed\n");
		cudaStreamDestroy(stream);
		cudaFree(d_buffer);
		return 1;
	}
	if (!recorder.register_region(0, d_buffer, size_bytes, 1)) {
		std::printf("register_region failed\n");
		recorder.shutdown();
		cudaStreamDestroy(stream);
		cudaFree(d_buffer);
		return 1;
	}

	bool ok = true;

	// Demo 1: off-by-one checkpoint bug (correctness + trace count)
	if (run_demo1) {
		tt::TraceCollector trace;
		const auto trace_start = std::chrono::steady_clock::now();

		int32_t remaining_work = static_cast<int32_t>(checkpoint_count);

		auto run_and_account = [&](uint32_t checkpoint) -> bool {
			if (!run_checkpoint(checkpoint, trace, recorder, stream, d_buffer, element_count, trace_start)) {
				recorder.shutdown();
				cudaStreamDestroy(stream);
				cudaFree(d_buffer);
				return false;
			}
			--remaining_work;
			return true;
			};

		if (bug_mode) {
			// BUG: off-by-one loop boundary.
			for (uint32_t checkpoint = 0; checkpoint <= checkpoint_count; ++checkpoint) {
				if (!run_and_account(checkpoint))
					return 1;
			}
		}
		else {
			for (uint32_t checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
				if (!run_and_account(checkpoint))
					return 1;
			}
		}

		// clean mode => 0, bug mode => -1
		std::printf("remaining_work=%d (expected 0)\n", remaining_work);

		if (!check_cuda(cudaStreamSynchronize(stream), "stream sync")) {
			recorder.shutdown();
			cudaStreamDestroy(stream);
			cudaFree(d_buffer);
			return 1;
		}

		trace.write(bug_mode ? "trace/tt_demo_single-bug.json" : "trace/tt_demo_single.json");

		std::vector<uint32_t> host_out(element_count);
		for (uint32_t checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
			if (!recorder.rewind_to_checkpoint(checkpoint, stream)) {
				std::printf("rewind_to_checkpoint failed at %u\n", checkpoint);
				ok = false;
				break;
			}
			if (!check_cuda(cudaMemcpy(host_out.data(), d_buffer, size_bytes,
				cudaMemcpyDeviceToHost), "memcpy out")) {
				ok = false;
				break;
			}
			if (!verify_pattern(host_out, checkpoint)) {
				std::printf("verify failed at checkpoint %u\n", checkpoint);
				ok = false;
				break;
			}
		}
	}

	// Demo 2: mysterious slowdown from '&' vs '&&' causing unconditional sync
	if (run_demo2) {
		tt::TraceCollector trace2;
		const auto trace2_start = std::chrono::steady_clock::now();

		if (!run_sync_demo(sync_bug, trace2, stream, trace2_start)) {
			std::printf("tt_demo_sync failed%s\n", sync_bug ? " (sync bug)" : "");
			recorder.shutdown();
			cudaStreamDestroy(stream);
			cudaFree(d_buffer);
			return 1;
		}

		trace2.write(sync_bug ? "trace/tt_demo_sync-bug.json"
			: "trace/tt_demo_sync.json");

		std::printf("tt_demo_sync wrote trace (%s)\n",
			sync_bug ? "sync bug" : "clean");
	}

	// Demo 3: deterministic time-travel demo: find first bad checkpoint from overflow/wrap
	if (run_demo3) {
		// Chosen so overflow occurs after several checkpoints ("good then bad forever").
		// 0x20000000 * 8 = 0x100000000 -> first bad checkpoint is expected around 8.
		const uint32_t checkpoint_count3 = 12;
		const uint32_t add_per_checkpoint = 0x20000000u;

		const auto trace3_start = std::chrono::steady_clock::now();

		uint32_t verified_checkpoints = 0;
		const bool demo3_ok = run_accum_overflow_demo(
			tt_debug,
			tt_show_list,
			stream, d_buffer, element_count, size_bytes,
			checkpoint_count3, add_per_checkpoint, trace3_start, &verified_checkpoints);
		if (!demo3_ok) {
			recorder.shutdown();
			cudaStreamDestroy(stream);
			cudaFree(d_buffer);
			return 1;
		}
	}

	recorder.shutdown();
	cudaStreamDestroy(stream);
	cudaFree(d_buffer);

	if (run_demo1 && !ok) {
		std::printf("tt_demo_single failed verification%s\n",
			bug_mode ? " (bug mode)" : "");
		return 1;
	}

	std::printf("tt_demo_single success\n");
	return 0;
}

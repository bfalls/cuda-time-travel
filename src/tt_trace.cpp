#include "tt/tt_trace.h"

#include <filesystem>
#include <fstream>

namespace tt {

namespace {

void write_escaped(std::ofstream& out, const std::string& value) {
    for (char c : value) {
        switch (c) {
        case '\\':
            out << "\\\\";
            break;
        case '\"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out << c;
            break;
        }
    }
}

} // namespace

void TraceCollector::add_event(const TraceEvent& event) {
    events_.push_back(event);
}

bool TraceCollector::write(const char* path) const {
    if (!path) {
        return false;
    }

    std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }

    out << "{ \"traceEvents\": [";
    for (size_t i = 0; i < events_.size(); ++i) {
        const TraceEvent& e = events_[i];
        if (i > 0) {
            out << ",";
        }
        out << "{";
        out << "\"name\":\"";
        write_escaped(out, e.name);
        out << "\",\"cat\":\"";
        write_escaped(out, e.cat);
        out << "\",\"ph\":\"X\"";
        out << ",\"ts\":" << e.ts_us;
        out << ",\"dur\":" << e.dur_us;
        out << ",\"pid\":" << e.pid;
        out << ",\"tid\":" << e.tid;
        out << ",\"args\":{";
        for (size_t a = 0; a < e.args.size(); ++a) {
            if (a > 0) {
                out << ",";
            }
            out << "\"";
            write_escaped(out, e.args[a].key);
            out << "\":";
            if (e.args[a].is_string) {
                out << "\"";
                write_escaped(out, e.args[a].value);
                out << "\"";
            } else {
                out << e.args[a].value;
            }
        }
        out << "}}";
    }
    out << "] }";
    return true;
}

} // namespace tt

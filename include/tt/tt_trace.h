#ifndef TT_TRACE_H
#define TT_TRACE_H

#include <string>
#include <vector>

namespace tt {

struct TraceArg {
    std::string key;
    std::string value;
    bool is_string = false;
};

struct TraceEvent {
    std::string name;
    std::string cat;
    double ts_us = 0.0;
    double dur_us = 0.0;
    int pid = 1;
    int tid = 0;
    std::vector<TraceArg> args;
};

class TraceCollector {
public:
    void add_event(const TraceEvent& event);
    bool write(const char* path) const;

private:
    std::vector<TraceEvent> events_;
};

} // namespace tt

#endif // TT_TRACE_H

#ifndef ORB_SLAM3_MAPPING_PROGRESS_H
#define ORB_SLAM3_MAPPING_PROGRESS_H

#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

// The mapping-progress contract is deliberately small and independent of any
// caller.  Consumers poll the caller-configured file and only need this one
// schema for a standalone trajectory or an Atlas trajectory.
//
// Writes use a same-directory temporary file followed by rename(2), so a
// reader observes either the previous complete document or the next one.
class MappingProgress
{
public:
    explicit MappingProgress(const std::string& path) : mPath(path), mSequence(0)
    {
        if(mPath.empty())
            throw std::runtime_error("mapping progress file path must not be empty");
    }

    void Write(const std::string& stage, std::size_t framesTotal, std::size_t framesProcessed)
    {
        if(!IsValidStage(stage))
            throw std::runtime_error("invalid mapping progress stage: " + stage);
        if(framesProcessed > framesTotal)
            throw std::runtime_error("mapping progress frames_processed exceeds frames_total");

        const std::string temporary = mPath + ".tmp." + std::to_string(static_cast<long long>(getpid()))
            + "." + std::to_string(mSequence++);
        std::ofstream output(temporary.c_str(), std::ios::out | std::ios::trunc);
        if(!output)
            throw std::runtime_error("cannot open mapping progress temporary file: " + temporary);

        output << "{\"schema_version\":1,\"stage\":\"" << stage
               << "\",\"frames_total\":" << framesTotal
               << ",\"frames_processed\":" << framesProcessed
               << ",\"updated_at_unix_ms\":" << UnixMilliseconds() << "}\n";
        output.close();
        if(!output)
        {
            std::remove(temporary.c_str());
            throw std::runtime_error("failed while writing mapping progress temporary file: " + temporary);
        }
        if(std::rename(temporary.c_str(), mPath.c_str()) != 0)
        {
            std::remove(temporary.c_str());
            throw std::runtime_error("cannot atomically replace mapping progress file: " + mPath);
        }
    }

private:
    static bool IsValidStage(const std::string& stage)
    {
        return stage == "loading" || stage == "tracking" || stage == "exporting"
            || stage == "completed" || stage == "failed"
            || stage == "atlas_loading" || stage == "atlas_tracking"
            || stage == "atlas_merging" || stage == "atlas_exporting";
    }

    static long long UnixMilliseconds()
    {
        const std::chrono::milliseconds now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch());
        return static_cast<long long>(now.count());
    }

    std::string mPath;
    std::size_t mSequence;
};

#endif

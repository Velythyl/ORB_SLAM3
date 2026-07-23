// Multi-session RGB-D Atlas runner for RAGMAP.
// Scene manifest rows: trajectory_id dataset_directory sequence_manifest
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <System.h>

using namespace std;

struct Frame
{
    double timestamp;
    string image;
    string depth;
};

struct Session
{
    string id;
    string dataset;
    string manifest;
    vector<Frame> frames;
};

static string trim(const string& value)
{
    const string whitespace = " \t\r\n";
    const size_t begin = value.find_first_not_of(whitespace);
    if(begin == string::npos)
        return "";
    return value.substr(begin, value.find_last_not_of(whitespace) - begin + 1);
}

static bool is_absolute_path(const string& path)
{
    return !path.empty() && path[0] == '/';
}

static string frame_path(const string& dataset, const string& path)
{
    if(is_absolute_path(path))
        return path;
    if(dataset.empty())
        return path;
    return dataset + "/" + path;
}

static string json_escape(const string& value)
{
    ostringstream out;
    for(const unsigned char ch : value)
    {
        switch(ch)
        {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if(ch < 0x20)
                out << "\\u" << hex << setw(4) << setfill('0') << static_cast<int>(ch) << dec << setfill(' ');
            else
                out << static_cast<char>(ch);
        }
    }
    return out.str();
}

static vector<Frame> load_frames(const string& manifest)
{
    ifstream input(manifest.c_str());
    if(!input)
        throw runtime_error("cannot read sequence manifest: " + manifest);

    vector<Frame> frames;
    string line;
    size_t line_number = 0;
    while(getline(input, line))
    {
        ++line_number;
        line = trim(line);
        if(line.empty() || line[0] == '#')
            continue;

        // Accept the compact format "rgb_timestamp rgb depth" and the
        // conventional TUM association format "rgb_timestamp rgb depth_timestamp depth".
        istringstream row(line);
        Frame frame;
        double depth_timestamp = 0.0;
        string extra;
        if(!(row >> frame.timestamp >> frame.image))
            throw runtime_error("invalid RGB-D manifest row " + to_string(line_number) + " in " + manifest);

        if(!(row >> frame.depth))
            throw runtime_error("RGB-D manifest row " + to_string(line_number) + " is missing a depth image in " + manifest);

        // A four-column association has the depth timestamp in column three.
        // Detect it without accepting arbitrary trailing columns.
        istringstream third_column(frame.depth);
        if((third_column >> depth_timestamp) && third_column.eof())
        {
            if(!(row >> frame.depth))
                throw runtime_error("RGB-D manifest row " + to_string(line_number) + " is missing a depth image in " + manifest);
        }
        if(row >> extra)
            throw runtime_error("invalid extra column in RGB-D manifest row " + to_string(line_number) + " in " + manifest);
        if(!isfinite(frame.timestamp) || !isfinite(depth_timestamp) || frame.image.empty() || frame.depth.empty())
            throw runtime_error("invalid RGB-D manifest row " + to_string(line_number) + " in " + manifest);
        frames.push_back(frame);
    }
    return frames;
}

static vector<Session> load_scene(const string& path)
{
    ifstream input(path.c_str());
    if(!input)
        throw runtime_error("cannot read scene manifest: " + path);

    vector<Session> sessions;
    set<string> ids;
    string line;
    size_t line_number = 0;
    while(getline(input, line))
    {
        ++line_number;
        line = trim(line);
        if(line.empty() || line[0] == '#')
            continue;

        istringstream row(line);
        Session session;
        string extra;
        if(!(row >> session.id >> session.dataset >> session.manifest) || (row >> extra))
            throw runtime_error("scene manifest row " + to_string(line_number) + " must be: trajectory_id dataset_directory sequence_manifest");
        if(!ids.insert(session.id).second)
            throw runtime_error("duplicate trajectory_id in scene manifest: " + session.id);
        session.frames = load_frames(session.manifest);
        if(session.frames.empty())
            throw runtime_error("trajectory has no RGB-D frames: " + session.id);
        sessions.push_back(session);
    }
    return sessions;
}

static void write_status(const string& path, size_t trajectory_count, int map_count, size_t tracked_frame_count, const string& pose_path)
{
    ofstream status(path.c_str());
    if(!status)
        throw runtime_error("cannot open Atlas status output: " + path);
    status << "{\"status\":\"completed\",\"trajectory_count\":" << trajectory_count
           << ",\"map_count\":" << map_count
           << ",\"merged_common_map\":" << (map_count == 1 ? "true" : "false")
           << ",\"tracked_frame_count\":" << tracked_frame_count
           << ",\"final_pose_output_path\":\"" << json_escape(pose_path) << "\"}\n";
    status.close();
    if(!status)
        throw runtime_error("failed while writing Atlas status output: " + path);
}

int main(int argc, char** argv)
{
    if(argc != 6)
    {
        cerr << "usage: scene_atlas_export VOCAB SETTINGS SCENE_MANIFEST FINAL_POSES_JSONL ATLAS_STATUS_JSON\n";
        return 2;
    }

    unique_ptr<ORB_SLAM3::System> slam;
    bool shut_down = false;
    try
    {
        const vector<Session> sessions = load_scene(argv[3]);
        if(sessions.empty())
            throw runtime_error("scene manifest contains no trajectories");

        slam.reset(new ORB_SLAM3::System(argv[1], argv[2], ORB_SLAM3::System::RGBD, false));
        for(size_t session_index = 0; session_index < sessions.size(); ++session_index)
        {
            if(session_index != 0)
                slam->ChangeDataset();

            const Session& session = sessions[session_index];
            for(size_t frame_index = 0; frame_index < session.frames.size(); ++frame_index)
            {
                const Frame& frame = session.frames[frame_index];
                const string image_path = frame_path(session.dataset, frame.image);
                const string depth_path = frame_path(session.dataset, frame.depth);
                const cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
                const cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
                if(image.empty())
                    throw runtime_error("cannot load RGB image for trajectory " + session.id + ": " + image_path);
                if(depth.empty())
                    throw runtime_error("cannot load depth image for trajectory " + session.id + ": " + depth_path);

                slam->TrackRGBD(image, depth, frame.timestamp, {}, frame.image, session.id, static_cast<long>(frame_index));
            }
        }

        // Shutdown waits for loop closing and global optimization before the
        // trajectory history is converted into final Atlas-frame poses.
        slam->Shutdown();
        shut_down = true;
        size_t tracked_frame_count = 0;
        if(!slam->SaveTrajectoryTUMWithMetadata(argv[4], &tracked_frame_count))
            throw runtime_error("failed to export final Atlas poses to " + string(argv[4]));
        if(tracked_frame_count == 0)
            throw runtime_error("no tracked frames were available for final Atlas pose export");

        const int map_count = slam->AtlasMapCount();
        write_status(argv[5], sessions.size(), map_count, tracked_frame_count, argv[4]);
        return 0;
    }
    catch(const exception& error)
    {
        if(slam && !shut_down)
            slam->Shutdown();
        cerr << "scene_atlas_export: " << error.what() << "\n";
        return 1;
    }
}

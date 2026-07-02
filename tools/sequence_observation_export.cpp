#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <MapPoint.h>
#include <System.h>

using namespace std;

struct ImageEntry {
    double timestamp = 0.0;
    string image;
    string right;
    string depth;
};

struct ExportedMapPoint {
    long unsigned int id = 0;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    int observations = 0;
    int found = 0;
    string descriptor_hex;
};

static vector<ImageEntry> load_mono_images(const string& dataset) {
    ifstream in(dataset + "/rgb.txt");
    if (!in) {
        throw runtime_error("failed to open " + dataset + "/rgb.txt");
    }

    vector<ImageEntry> entries;
    string line;
    while (getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        istringstream ss(line);
        ImageEntry entry;
        if (ss >> entry.timestamp >> entry.image) {
            entries.push_back(entry);
        }
    }
    return entries;
}

static vector<ImageEntry> load_rgbd_images(const string& association) {
    ifstream in(association);
    if (!in) {
        throw runtime_error("failed to open " + association);
    }

    vector<ImageEntry> entries;
    string line;
    while (getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        double depth_timestamp = 0.0;
        ImageEntry entry;
        istringstream ss(line);
        if (ss >> entry.timestamp >> entry.image >> depth_timestamp >> entry.depth) {
            entries.push_back(entry);
        }
    }
    return entries;
}

static vector<ImageEntry> load_manifest_images(const string& manifest, const string& sensor) {
    ifstream in(manifest);
    if (!in) {
        throw runtime_error("failed to open " + manifest);
    }

    vector<ImageEntry> entries;
    string line;
    int line_number = 0;
    while (getline(in, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        ImageEntry entry;
        istringstream ss(line);
        if (!(ss >> entry.timestamp >> entry.image)) {
            throw runtime_error("invalid manifest row " + to_string(line_number) + " in " + manifest);
        }
        if (sensor == "stereo") {
            ss >> entry.right;
        } else {
            ss >> entry.depth;
        }
        if (sensor == "stereo" && entry.right.empty()) {
            throw runtime_error("manifest row " + to_string(line_number) + " is missing a right image");
        }
        if (sensor == "rgbd" && entry.depth.empty()) {
            throw runtime_error("manifest row " + to_string(line_number) + " is missing a depth image");
        }
        if (sensor != "rgbd") {
            entry.depth.clear();
        }
        if (sensor != "stereo") {
            entry.right.clear();
        }
        entries.push_back(entry);
    }
    return entries;
}

static string json_escape(const string& value) {
    ostringstream out;
    for (const char ch : value) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                out << "\\u" << hex << setw(4) << setfill('0') << static_cast<int>(ch);
            } else {
                out << ch;
            }
        }
    }
    return out.str();
}

static string descriptor_hex(const cv::Mat& descriptor) {
    if (descriptor.empty()) {
        return "";
    }
    const cv::Mat row = descriptor.reshape(1, 1);
    ostringstream out;
    out << hex << setfill('0');
    for (int i = 0; i < row.cols; ++i) {
        out << setw(2) << static_cast<int>(row.at<unsigned char>(0, i));
    }
    return out.str();
}

static void remember_map_point(ORB_SLAM3::MapPoint* point, map<long unsigned int, ExportedMapPoint>& out) {
    if (!point || point->isBad()) {
        return;
    }
    const long unsigned int id = point->mnId;
    if (out.find(id) != out.end()) {
        return;
    }

    const Eigen::Vector3f position = point->GetWorldPos();
    ExportedMapPoint exported;
    exported.id = id;
    exported.x = position[0];
    exported.y = position[1];
    exported.z = position[2];
    exported.observations = point->Observations();
    exported.found = point->GetFound();
    exported.descriptor_hex = descriptor_hex(point->GetDescriptor());
    out[id] = exported;
}

static void write_pose_json(ostream& out, const Sophus::SE3f& Tcw) {
    const Sophus::SE3f Twc = Tcw.inverse();
    const Eigen::Vector3f t = Twc.translation();
    const Eigen::Quaternionf q = Twc.unit_quaternion();
    out << "{\"tx\":" << t[0]
        << ",\"ty\":" << t[1]
        << ",\"tz\":" << t[2]
        << ",\"qx\":" << q.x()
        << ",\"qy\":" << q.y()
        << ",\"qz\":" << q.z()
        << ",\"qw\":" << q.w()
        << "}";
}

static bool pose_is_valid(const Sophus::SE3f& pose, int tracking_state) {
    return (tracking_state == 2 || tracking_state == 3 || tracking_state == 5) && !pose.matrix().hasNaN();
}

static void write_observation(
    ostream& out,
    int frame_index,
    const ImageEntry& entry,
    int tracking_state,
    const Sophus::SE3f& Tcw,
    ORB_SLAM3::System& slam,
    map<long unsigned int, ExportedMapPoint>& map_points) {
    const vector<cv::KeyPoint> keypoints = slam.GetTrackedKeyPointsUn();
    const vector<ORB_SLAM3::MapPoint*> tracked_points = slam.GetTrackedMapPoints();

    out << fixed << setprecision(9);
    out << "{\"frame_index\":" << frame_index
        << ",\"timestamp\":" << entry.timestamp
        << ",\"image\":\"" << json_escape(entry.image) << "\"";
    if (entry.right.empty()) {
        out << ",\"right\":null";
    } else {
        out << ",\"right\":\"" << json_escape(entry.right) << "\"";
    }
    if (entry.depth.empty()) {
        out << ",\"depth\":null";
    } else {
        out << ",\"depth\":\"" << json_escape(entry.depth) << "\"";
    }
    out << ",\"tracking_state\":" << tracking_state << ",\"pose\":";
    if (pose_is_valid(Tcw, tracking_state)) {
        write_pose_json(out, Tcw);
    } else {
        out << "null";
    }

    out << ",\"features\":[";
    for (size_t i = 0; i < keypoints.size(); ++i) {
        ORB_SLAM3::MapPoint* point = i < tracked_points.size() ? tracked_points[i] : nullptr;
        const bool has_point = point && !point->isBad();
        if (has_point) {
            remember_map_point(point, map_points);
        }

        const cv::KeyPoint& kp = keypoints[i];
        if (i > 0) {
            out << ",";
        }
        out << "{\"index\":" << i
            << ",\"x\":" << kp.pt.x
            << ",\"y\":" << kp.pt.y
            << ",\"size\":" << kp.size
            << ",\"angle\":" << kp.angle
            << ",\"response\":" << kp.response
            << ",\"octave\":" << kp.octave
            << ",\"class_id\":" << kp.class_id
            << ",\"map_point_id\":";
        if (has_point) {
            out << point->mnId;
        } else {
            out << "null";
        }
        out << "}";
    }
    out << "]}\n";
}

static void write_map_points_csv(const string& path, const map<long unsigned int, ExportedMapPoint>& points) {
    ofstream out(path);
    if (!out) {
        throw runtime_error("failed to open " + path);
    }
    out << "id,x,y,z,observations,found,descriptor_hex\n";
    out << fixed << setprecision(9);
    for (const auto& item : points) {
        const ExportedMapPoint& point = item.second;
        out << point.id << ","
            << point.x << ","
            << point.y << ","
            << point.z << ","
            << point.observations << ","
            << point.found << ","
            << point.descriptor_hex << "\n";
    }
}

static void save_trajectories(ORB_SLAM3::System& slam, const string& sensor) {
    if (sensor == "rgbd" || sensor == "stereo") {
        slam.SaveTrajectoryTUM("CameraTrajectory.txt");
    }
    slam.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

static ORB_SLAM3::System::eSensor orb_sensor(const string& sensor) {
    if (sensor == "monocular") {
        return ORB_SLAM3::System::MONOCULAR;
    }
    if (sensor == "stereo") {
        return ORB_SLAM3::System::STEREO;
    }
    if (sensor == "rgbd") {
        return ORB_SLAM3::System::RGBD;
    }
    throw runtime_error("unsupported sensor for observation export: " + sensor);
}

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8 && argc != 9) {
        cerr << "usage: sequence_observation_export SENSOR VOCAB SETTINGS DATASET OBSERVATIONS_JSONL MAP_POINTS_CSV [ASSOCIATION]\n";
        cerr << "       sequence_observation_export SENSOR VOCAB SETTINGS DATASET OBSERVATIONS_JSONL MAP_POINTS_CSV --manifest MANIFEST\n";
        return 2;
    }

    const string sensor = argv[1];
    const string vocabulary = argv[2];
    const string settings = argv[3];
    const string dataset = argv[4];
    const string observations_path = argv[5];
    const string map_points_path = argv[6];
    string association;
    string manifest;
    if (argc == 8) {
        association = argv[7];
    } else if (argc == 9) {
        const string flag = argv[7];
        if (flag != "--manifest") {
            cerr << "error: expected --manifest before manifest path\n";
            return 2;
        }
        manifest = argv[8];
    }

    try {
        const bool rgbd = sensor == "rgbd";
        const bool stereo = sensor == "stereo";
        const ORB_SLAM3::System::eSensor system_sensor = orb_sensor(sensor);

        if (stereo && manifest.empty()) {
            throw runtime_error("stereo runs require --manifest with timestamp left_image right_image rows");
        }

        vector<ImageEntry> entries;
        if (!manifest.empty()) {
            entries = load_manifest_images(manifest, sensor);
        } else {
            entries = rgbd ? load_rgbd_images(association) : load_mono_images(dataset);
        }
        if (entries.empty()) {
            throw runtime_error("no images found");
        }

        ORB_SLAM3::System slam(vocabulary, settings, system_sensor, false);
        const float image_scale = slam.GetImageScale();

        ofstream observations(observations_path);
        if (!observations) {
            throw runtime_error("failed to open " + observations_path);
        }

        map<long unsigned int, ExportedMapPoint> map_points;
        vector<float> tracking_times(entries.size());

        cout << endl << "-------" << endl;
        cout << "Start processing sequence with observation export ..." << endl;
        cout << "Images in the sequence: " << entries.size() << endl << endl;

        for (size_t i = 0; i < entries.size(); ++i) {
            const ImageEntry& entry = entries[i];
            cv::Mat image = cv::imread(dataset + "/" + entry.image, cv::IMREAD_UNCHANGED);
            if (image.empty()) {
                throw runtime_error("failed to load image " + dataset + "/" + entry.image);
            }

            cv::Mat right;
            if (stereo) {
                right = cv::imread(dataset + "/" + entry.right, cv::IMREAD_UNCHANGED);
                if (right.empty()) {
                    throw runtime_error("failed to load right image " + dataset + "/" + entry.right);
                }
            }

            cv::Mat depth;
            if (rgbd) {
                depth = cv::imread(dataset + "/" + entry.depth, cv::IMREAD_UNCHANGED);
                if (depth.empty()) {
                    throw runtime_error("failed to load depth image " + dataset + "/" + entry.depth);
                }
            }

            if (image_scale != 1.0f) {
                const int width = static_cast<int>(image.cols * image_scale);
                const int height = static_cast<int>(image.rows * image_scale);
                cv::resize(image, image, cv::Size(width, height));
                if (stereo) {
                    cv::resize(right, right, cv::Size(width, height));
                }
                if (rgbd) {
                    cv::resize(depth, depth, cv::Size(width, height));
                }
            }

            const auto start = chrono::steady_clock::now();
            Sophus::SE3f Tcw;
            if (rgbd) {
                Tcw = slam.TrackRGBD(image, depth, entry.timestamp);
            } else if (stereo) {
                Tcw = slam.TrackStereo(image, right, entry.timestamp);
            } else {
                Tcw = slam.TrackMonocular(image, entry.timestamp);
            }
            const auto end = chrono::steady_clock::now();
            tracking_times[i] = chrono::duration_cast<chrono::duration<float> >(end - start).count();

            write_observation(observations, static_cast<int>(i), entry, slam.GetTrackingState(), Tcw, slam, map_points);

            double wait_time = 0.0;
            if (i + 1 < entries.size()) {
                wait_time = entries[i + 1].timestamp - entry.timestamp;
            } else if (i > 0) {
                wait_time = entry.timestamp - entries[i - 1].timestamp;
            }
            if (tracking_times[i] < wait_time) {
                usleep((wait_time - tracking_times[i]) * 1e6);
            }
        }

        slam.Shutdown();

        sort(tracking_times.begin(), tracking_times.end());
        float total_time = 0.0f;
        for (float time : tracking_times) {
            total_time += time;
        }
        cout << "-------" << endl << endl;
        cout << "median tracking time: " << tracking_times[tracking_times.size() / 2] << endl;
        cout << "mean tracking time: " << total_time / tracking_times.size() << endl;

        save_trajectories(slam, sensor);
        write_map_points_csv(map_points_path, map_points);

        cout << "wrote " << entries.size() << " posed observations to " << observations_path << endl;
        cout << "wrote " << map_points.size() << " observed map points to " << map_points_path << endl;
        return 0;
    } catch (const exception& exc) {
        cerr << "error: " << exc.what() << endl;
        return 1;
    }
}

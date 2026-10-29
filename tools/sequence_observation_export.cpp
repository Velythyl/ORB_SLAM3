#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
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

struct ExportedFeature {
    float x = 0.0f;
    float y = 0.0f;
    float size = 0.0f;
    float angle = 0.0f;
    float response = 0.0f;
    int octave = 0;
    int class_id = 0;
    bool has_map_point = false;
    long unsigned int map_point_id = 0;
};

struct ExportedObservation {
    int frame_index = 0;
    ImageEntry entry;
    int tracking_state = 0;
    Sophus::SE3f Tcw;
    vector<ExportedFeature> features;
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

struct OdometryRow {
    double timestamp = 0.0;
    Sophus::SE3f Tob;
};

// TUM rows "t x y z qx qy qz qw": base pose in the odometry frame.
static vector<OdometryRow> load_odometry(const string& path) {
    ifstream in(path);
    if (!in) {
        throw runtime_error("failed to open odometry file " + path);
    }
    vector<OdometryRow> rows;
    string line;
    int line_number = 0;
    while (getline(in, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        istringstream ss(line);
        double t, x, y, z, qx, qy, qz, qw;
        if (!(ss >> t >> x >> y >> z >> qx >> qy >> qz >> qw)) {
            throw runtime_error("invalid odometry row " + to_string(line_number) + " in " + path);
        }
        Eigen::Quaterniond q(qw, qx, qy, qz);
        if (q.norm() < 1e-6) {
            throw runtime_error("zero quaternion in odometry row " + to_string(line_number));
        }
        q.normalize();
        OdometryRow row;
        row.timestamp = t;
        row.Tob = Sophus::SE3f(q.cast<float>(), Eigen::Vector3f(x, y, z));
        rows.push_back(row);
    }
    sort(rows.begin(), rows.end(), [](const OdometryRow& a, const OdometryRow& b) { return a.timestamp < b.timestamp; });
    return rows;
}

// Rows are already interpolated to the colour stamps; match within 1 ms.
static const OdometryRow* find_odometry(const vector<OdometryRow>& rows, double timestamp) {
    const double tolerance = 1e-3;
    auto it = lower_bound(rows.begin(), rows.end(), timestamp,
                          [](const OdometryRow& row, double t) { return row.timestamp < t; });
    const OdometryRow* best = nullptr;
    double best_dt = tolerance;
    if (it != rows.end() && fabs(it->timestamp - timestamp) <= best_dt) {
        best = &*it;
        best_dt = fabs(it->timestamp - timestamp);
    }
    if (it != rows.begin()) {
        --it;
        if (fabs(it->timestamp - timestamp) <= best_dt) {
            best = &*it;
        }
    }
    return best;
}

// {"T_base_camera": [[...],[...],[...],[...]]} (or 16 flat numbers), row-major,
// camera optical frame -> base frame.
static Sophus::SE3f load_base_to_camera(const string& path) {
    ifstream in(path);
    if (!in) {
        throw runtime_error("failed to open " + path);
    }
    const string text((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    const string key = "\"T_base_camera\"";
    size_t pos = text.find(key);
    if (pos == string::npos) {
        throw runtime_error(path + " has no \"T_base_camera\" key");
    }
    pos = text.find('[', pos + key.size());
    if (pos == string::npos) {
        throw runtime_error("T_base_camera in " + path + " is not an array");
    }
    vector<double> values;
    int depth = 0;
    const char* begin = text.c_str();
    for (size_t i = pos; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch == '[') {
            ++depth;
        } else if (ch == ']') {
            if (--depth == 0) {
                break;
            }
        } else if (ch == '-' || ch == '+' || ch == '.' || isdigit(static_cast<unsigned char>(ch))) {
            char* end = nullptr;
            const double value = strtod(begin + i, &end);
            if (end == begin + i) {
                throw runtime_error("unparsable number in T_base_camera of " + path);
            }
            values.push_back(value);
            i = static_cast<size_t>(end - begin) - 1;
        }
    }
    if (values.size() != 16) {
        throw runtime_error("T_base_camera in " + path + " must hold 16 numbers, found " + to_string(values.size()));
    }
    Eigen::Matrix4d T;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            T(r, c) = values[4 * r + c];
        }
    }
    if ((T.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm() > 1e-6) {
        throw runtime_error("T_base_camera in " + path + " must end with the row [0, 0, 0, 1]");
    }
    const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    if ((R.transpose() * R - Eigen::Matrix3d::Identity()).norm() > 1e-3 || R.determinant() < 0) {
        throw runtime_error("T_base_camera in " + path + " does not hold a rotation");
    }
    Eigen::Quaterniond q(R);
    q.normalize();
    return Sophus::SE3f(q.cast<float>(), T.block<3, 1>(0, 3).cast<float>());
}

static float parse_positive(const string& flag, const char* value) {
    char* end = nullptr;
    const float parsed = strtof(value, &end);
    if (end == value || *end != '\0' || !(parsed > 0.0f)) {
        throw runtime_error(flag + " requires a positive number");
    }
    return parsed;
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

static ExportedObservation capture_observation(
    int frame_index,
    const ImageEntry& entry,
    int tracking_state,
    const Sophus::SE3f& Tcw,
    ORB_SLAM3::System& slam,
    map<long unsigned int, ExportedMapPoint>& map_points) {
    const vector<cv::KeyPoint> keypoints = slam.GetTrackedKeyPointsUn();
    const vector<ORB_SLAM3::MapPoint*> tracked_points = slam.GetTrackedMapPoints();

    ExportedObservation observation;
    observation.frame_index = frame_index;
    observation.entry = entry;
    observation.tracking_state = tracking_state;
    observation.Tcw = Tcw;
    observation.features.reserve(keypoints.size());

    for (size_t i = 0; i < keypoints.size(); ++i) {
        ORB_SLAM3::MapPoint* point = i < tracked_points.size() ? tracked_points[i] : nullptr;
        const bool has_point = point && !point->isBad();
        if (has_point) {
            remember_map_point(point, map_points);
        }

        const cv::KeyPoint& kp = keypoints[i];
        ExportedFeature feature;
        feature.x = kp.pt.x;
        feature.y = kp.pt.y;
        feature.size = kp.size;
        feature.angle = kp.angle;
        feature.response = kp.response;
        feature.octave = kp.octave;
        feature.class_id = kp.class_id;
        feature.has_map_point = has_point;
        if (has_point) {
            feature.map_point_id = point->mnId;
        }
        observation.features.push_back(feature);
    }

    return observation;
}

static void write_observation(ostream& out, const ExportedObservation& observation) {
    out << fixed << setprecision(9);
    out << "{\"frame_index\":" << observation.frame_index
        << ",\"timestamp\":" << observation.entry.timestamp
        << ",\"image\":\"" << json_escape(observation.entry.image) << "\"";
    if (observation.entry.right.empty()) {
        out << ",\"right\":null";
    } else {
        out << ",\"right\":\"" << json_escape(observation.entry.right) << "\"";
    }
    if (observation.entry.depth.empty()) {
        out << ",\"depth\":null";
    } else {
        out << ",\"depth\":\"" << json_escape(observation.entry.depth) << "\"";
    }
    out << ",\"tracking_state\":" << observation.tracking_state << ",\"pose\":";
    if (pose_is_valid(observation.Tcw, observation.tracking_state)) {
        write_pose_json(out, observation.Tcw);
    } else {
        out << "null";
    }

    out << ",\"features\":[";
    for (size_t i = 0; i < observation.features.size(); ++i) {
        const ExportedFeature& feature = observation.features[i];
        if (i > 0) {
            out << ",";
        }
        out << "{\"index\":" << i
            << ",\"x\":" << feature.x
            << ",\"y\":" << feature.y
            << ",\"size\":" << feature.size
            << ",\"angle\":" << feature.angle
            << ",\"response\":" << feature.response
            << ",\"octave\":" << feature.octave
            << ",\"class_id\":" << feature.class_id
            << ",\"map_point_id\":";
        if (feature.has_map_point) {
            out << feature.map_point_id;
        } else {
            out << "null";
        }
        out << "}";
    }
    out << "]}\n";
}

class ObservationWriter {
public:
    explicit ObservationWriter(ostream& out) : out_(out), thread_(&ObservationWriter::Run, this) {}

    ~ObservationWriter() {
        Close();
    }

    void Push(ExportedObservation observation) {
        unique_lock<mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < kMaxPendingObservations || closed_; });
        if (closed_) {
            throw runtime_error("cannot add an observation after the writer is closed");
        }
        queue_.push_back(std::move(observation));
        not_empty_.notify_one();
    }

    void Close() {
        {
            lock_guard<mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    static const size_t kMaxPendingObservations = 32;

    void Run() {
        while (true) {
            ExportedObservation observation;
            {
                unique_lock<mutex> lock(mutex_);
                not_empty_.wait(lock, [this] { return closed_ || !queue_.empty(); });
                if (queue_.empty()) {
                    return;
                }
                observation = std::move(queue_.front());
                queue_.pop_front();
                not_full_.notify_one();
            }
            write_observation(out_, observation);
        }
    }

    ostream& out_;
    deque<ExportedObservation> queue_;
    mutex mutex_;
    condition_variable not_empty_;
    condition_variable not_full_;
    bool closed_ = false;
    thread thread_;
};

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
    if (argc < 7) {
        cerr << "usage: sequence_observation_export SENSOR VOCAB SETTINGS DATASET OBSERVATIONS_JSONL MAP_POINTS_CSV [ASSOCIATION]\n";
        cerr << "       sequence_observation_export SENSOR VOCAB SETTINGS DATASET OBSERVATIONS_JSONL MAP_POINTS_CSV [--manifest MANIFEST] [--no-realtime] [--sync-export]\n";
        cerr << "           [--odometry TUM_FILE --base-to-camera JSON] [--odom-trans-sigma S] [--odom-rot-sigma S] [--no-odom-static-mask]\n";
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
    bool realtime = true;
    bool async_export = true;
    string odometry_path;
    string base_to_camera_path;
    ORB_SLAM3::OdometryParams odometry_params;
    for (int arg = 7; arg < argc; ++arg) {
        const string value = argv[arg];
        if (value == "--manifest") {
            if (++arg == argc) {
                cerr << "error: --manifest requires a path\n";
                return 2;
            }
            manifest = argv[arg];
        } else if (value == "--odometry" || value == "--base-to-camera" || value == "--odom-trans-sigma" || value == "--odom-rot-sigma") {
            if (++arg == argc) {
                cerr << "error: " << value << " requires a value\n";
                return 2;
            }
            try {
                if (value == "--odometry") {
                    odometry_path = argv[arg];
                } else if (value == "--base-to-camera") {
                    base_to_camera_path = argv[arg];
                } else if (value == "--odom-trans-sigma") {
                    odometry_params.transSigma = parse_positive(value, argv[arg]);
                } else {
                    odometry_params.rotSigma = parse_positive(value, argv[arg]);
                }
            } catch (const exception& exc) {
                cerr << "error: " << exc.what() << "\n";
                return 2;
            }
        } else if (value == "--no-odom-static-mask") {
            odometry_params.staticMask = false;
        } else if (value == "--no-realtime") {
            realtime = false;
        } else if (value == "--sync-export") {
            async_export = false;
        } else if (association.empty()) {
            association = value;
        } else {
            cerr << "error: unexpected argument " << value << "\n";
            return 2;
        }
    }

    try {
        const bool rgbd = sensor == "rgbd";
        const bool stereo = sensor == "stereo";
        const ORB_SLAM3::System::eSensor system_sensor = orb_sensor(sensor);

        if (stereo && manifest.empty()) {
            throw runtime_error("stereo runs require --manifest with timestamp left_image right_image rows");
        }
        const bool use_odometry = !odometry_path.empty();
        if (use_odometry != !base_to_camera_path.empty()) {
            throw runtime_error("--odometry and --base-to-camera must be given together");
        }
        if (use_odometry && !rgbd) {
            throw runtime_error("--odometry is only supported for the rgbd sensor");
        }
        vector<OdometryRow> odometry;
        Sophus::SE3f T_base_camera;
        if (use_odometry) {
            odometry = load_odometry(odometry_path);
            T_base_camera = load_base_to_camera(base_to_camera_path);
            if (odometry.empty()) {
                throw runtime_error("no odometry rows in " + odometry_path);
            }
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
        size_t frames_with_odometry = 0;
        if (use_odometry) {
            slam.EnableOdometry(T_base_camera, odometry_params);
        }

        ofstream observations(observations_path);
        if (!observations) {
            throw runtime_error("failed to open " + observations_path);
        }
        unique_ptr<ObservationWriter> observation_writer;
        if (async_export) {
            observation_writer.reset(new ObservationWriter(observations));
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
                if (use_odometry) {
                    const OdometryRow* row = find_odometry(odometry, entry.timestamp);
                    slam.SetFrameOdometry(row != nullptr, row ? row->Tob : Sophus::SE3f());
                    if (row) {
                        ++frames_with_odometry;
                    }
                }
                Tcw = slam.TrackRGBD(image, depth, entry.timestamp);
            } else if (stereo) {
                Tcw = slam.TrackStereo(image, right, entry.timestamp);
            } else {
                Tcw = slam.TrackMonocular(image, entry.timestamp);
            }
            const auto end = chrono::steady_clock::now();
            tracking_times[i] = chrono::duration_cast<chrono::duration<float> >(end - start).count();

            ExportedObservation observation = capture_observation(
                static_cast<int>(i), entry, slam.GetTrackingState(), Tcw, slam, map_points);
            if (observation_writer) {
                observation_writer->Push(std::move(observation));
            } else {
                write_observation(observations, observation);
            }

            if (realtime) {
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
        }

        if (observation_writer) {
            observation_writer->Close();
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
        if (use_odometry) {
            cout << "frames matched to odometry: " << frames_with_odometry << " of " << entries.size() << endl;
            slam.PrintOdometryStats();
        }

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

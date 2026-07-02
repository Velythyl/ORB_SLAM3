#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

struct ImageEntry {
    double t = 0.0;
    std::string path;
};

struct RgbdEntry {
    double t = 0.0;
    std::string color_path;
    std::string depth_path;
};

struct Camera {
    float fx = 517.306408f;
    float fy = 516.469215f;
    float cx = 318.643040f;
    float cy = 255.313989f;
    float depth_scale = 5000.0f;
};

struct Pose {
    double t = 0.0;
    cv::Matx33f r;
    cv::Vec3f p;
};

struct Point {
    float x;
    float y;
    float z;
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

static std::vector<ImageEntry> read_image_list(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }

    std::vector<ImageEntry> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ss(line);
        ImageEntry e;
        if (ss >> e.t >> e.path) {
            out.push_back(e);
        }
    }
    return out;
}

static std::vector<RgbdEntry> read_manifest(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }

    std::vector<RgbdEntry> out;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ss(line);
        RgbdEntry entry;
        if (!(ss >> entry.t >> entry.color_path >> entry.depth_path)) {
            throw std::runtime_error("invalid RGB-D manifest row " + std::to_string(line_number) + " in " + path);
        }
        out.push_back(entry);
    }
    return out;
}

static cv::Matx33f quat_to_mat(float x, float y, float z, float w) {
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;

    return cv::Matx33f(
        1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz),        2.0f * (xz + wy),
        2.0f * (xy + wz),        1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx),
        2.0f * (xz - wy),        2.0f * (yz + wx),        1.0f - 2.0f * (xx + yy));
}

static std::vector<Pose> read_poses(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }

    std::vector<Pose> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ss(line);
        float tx, ty, tz, qx, qy, qz, qw;
        Pose pose;
        if (ss >> pose.t >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
            pose.p = cv::Vec3f(tx, ty, tz);
            pose.r = quat_to_mat(qx, qy, qz, qw);
            out.push_back(pose);
        }
    }
    return out;
}

static int nearest_index(const std::vector<ImageEntry>& entries, double t) {
    auto it = std::lower_bound(entries.begin(), entries.end(), t,
        [](const ImageEntry& e, double value) { return e.t < value; });
    if (it == entries.begin()) {
        return 0;
    }
    if (it == entries.end()) {
        return static_cast<int>(entries.size() - 1);
    }
    const int hi = static_cast<int>(it - entries.begin());
    const int lo = hi - 1;
    return std::abs(entries[lo].t - t) <= std::abs(entries[hi].t - t) ? lo : hi;
}

static int nearest_rgbd_index(const std::vector<RgbdEntry>& entries, double t) {
    auto it = std::lower_bound(entries.begin(), entries.end(), t,
        [](const RgbdEntry& e, double value) { return e.t < value; });
    if (it == entries.begin()) {
        return 0;
    }
    if (it == entries.end()) {
        return static_cast<int>(entries.size() - 1);
    }
    const int hi = static_cast<int>(it - entries.begin());
    const int lo = hi - 1;
    return std::abs(entries[lo].t - t) <= std::abs(entries[hi].t - t) ? lo : hi;
}

static bool read_arg(int argc, char** argv, int& index, const std::string& name, std::string& out) {
    if (std::string(argv[index]) != name) {
        return false;
    }
    if (index + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
    }
    out = argv[++index];
    return true;
}

static bool read_arg(int argc, char** argv, int& index, const std::string& name, float& out) {
    std::string value;
    if (!read_arg(argc, argv, index, name, value)) {
        return false;
    }
    out = std::stof(value);
    return true;
}

static bool read_arg(int argc, char** argv, int& index, const std::string& name, int& out) {
    std::string value;
    if (!read_arg(argc, argv, index, name, value)) {
        return false;
    }
    out = std::atoi(value.c_str());
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: rgbd_keyframes_to_ply DATASET_DIR KEYFRAME_TRAJECTORY OUT.ply [options]\n";
        std::cerr << "options: --manifest PATH --fx N --fy N --cx N --cy N --depth-scale N";
        std::cerr << " --stride N --min-depth N --max-depth N\n";
        return 2;
    }

    const std::string dataset = argv[1];
    const std::string trajectory = argv[2];
    const std::string out_path = argv[3];

    std::string manifest;
    Camera camera;
    int stride = 4;
    float min_depth = 0.25f;
    float max_depth = 4.5f;

    try {
        for (int i = 4; i < argc; ++i) {
            if (read_arg(argc, argv, i, "--manifest", manifest)
                || read_arg(argc, argv, i, "--fx", camera.fx)
                || read_arg(argc, argv, i, "--fy", camera.fy)
                || read_arg(argc, argv, i, "--cx", camera.cx)
                || read_arg(argc, argv, i, "--cy", camera.cy)
                || read_arg(argc, argv, i, "--depth-scale", camera.depth_scale)
                || read_arg(argc, argv, i, "--stride", stride)
                || read_arg(argc, argv, i, "--min-depth", min_depth)
                || read_arg(argc, argv, i, "--max-depth", max_depth)) {
                continue;
            }
            throw std::runtime_error("unknown option " + std::string(argv[i]));
        }

        const auto rgb = manifest.empty() ? read_image_list(dataset + "/rgb.txt") : std::vector<ImageEntry>();
        const auto depth = manifest.empty() ? read_image_list(dataset + "/depth.txt") : std::vector<ImageEntry>();
        const auto rgbd = manifest.empty() ? std::vector<RgbdEntry>() : read_manifest(manifest);
        const auto poses = read_poses(trajectory);
        if (manifest.empty()) {
            if (rgb.empty()) {
                throw std::runtime_error("no RGB images found in " + dataset + "/rgb.txt");
            }
            if (depth.empty()) {
                throw std::runtime_error("no depth images found in " + dataset + "/depth.txt");
            }
        } else if (rgbd.empty()) {
            throw std::runtime_error("no RGB-D frames found in manifest " + manifest);
        }

        std::vector<Point> points;
        points.reserve(poses.size() * 16000);

        for (const auto& pose : poses) {
            std::string color_path;
            std::string depth_path;
            if (manifest.empty()) {
                const int rgb_i = nearest_index(rgb, pose.t);
                const int depth_i = nearest_index(depth, pose.t);
                if (std::abs(rgb[rgb_i].t - pose.t) > 0.05 || std::abs(depth[depth_i].t - pose.t) > 0.05) {
                    continue;
                }
                color_path = rgb[rgb_i].path;
                depth_path = depth[depth_i].path;
            } else {
                const int rgbd_i = nearest_rgbd_index(rgbd, pose.t);
                if (std::abs(rgbd[rgbd_i].t - pose.t) > 0.05) {
                    continue;
                }
                color_path = rgbd[rgbd_i].color_path;
                depth_path = rgbd[rgbd_i].depth_path;
            }

            const cv::Mat color = cv::imread(dataset + "/" + color_path, cv::IMREAD_COLOR);
            const cv::Mat z_image = cv::imread(dataset + "/" + depth_path, cv::IMREAD_UNCHANGED);
            if (color.empty() || z_image.empty() || z_image.type() != CV_16UC1) {
                std::cerr << "skipping unreadable frame near " << pose.t << "\n";
                continue;
            }

            for (int v = 0; v < z_image.rows; v += stride) {
                const uint16_t* zrow = z_image.ptr<uint16_t>(v);
                const cv::Vec3b* crow = color.ptr<cv::Vec3b>(v);
                for (int u = 0; u < z_image.cols; u += stride) {
                    const float z = static_cast<float>(zrow[u]) / camera.depth_scale;
                    if (z < min_depth || z > max_depth) {
                        continue;
                    }

                    const cv::Vec3f pc((u - camera.cx) * z / camera.fx, (v - camera.cy) * z / camera.fy, z);
                    const cv::Vec3f pw = pose.r * pc + pose.p;
                    const cv::Vec3b bgr = crow[u];
                    points.push_back(Point{pw[0], pw[1], pw[2], bgr[2], bgr[1], bgr[0]});
                }
            }
        }

        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("failed to open " + out_path);
        }

        out << "ply\n";
        out << "format binary_little_endian 1.0\n";
        out << "element vertex " << points.size() << "\n";
        out << "property float x\n";
        out << "property float y\n";
        out << "property float z\n";
        out << "property uchar red\n";
        out << "property uchar green\n";
        out << "property uchar blue\n";
        out << "end_header\n";
        for (const Point& p : points) {
            out.write(reinterpret_cast<const char*>(&p.x), sizeof(p.x));
            out.write(reinterpret_cast<const char*>(&p.y), sizeof(p.y));
            out.write(reinterpret_cast<const char*>(&p.z), sizeof(p.z));
            out.write(reinterpret_cast<const char*>(&p.r), sizeof(p.r));
            out.write(reinterpret_cast<const char*>(&p.g), sizeof(p.g));
            out.write(reinterpret_cast<const char*>(&p.b), sizeof(p.b));
        }

        std::cout << "wrote " << points.size() << " points to " << out_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

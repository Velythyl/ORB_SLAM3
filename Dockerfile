FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ORB_SLAM3_ROOT=/opt/orbslam3
ARG BUILD_JOBS=2
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        cmake \
        git \
        make \
        g++ \
        pkg-config \
        libboost-serialization-dev \
        libeigen3-dev \
        libgl1-mesa-dev \
        libglew-dev \
        libopencv-dev \
        libpython3-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git /tmp/Pangolin \
    && cmake -S /tmp/Pangolin -B /tmp/Pangolin/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/Pangolin/build --parallel ${BUILD_JOBS} \
    && cmake --install /tmp/Pangolin/build \
    && rm -rf /tmp/Pangolin \
    && ldconfig

COPY . ${ORB_SLAM3_ROOT}
WORKDIR ${ORB_SLAM3_ROOT}

RUN chmod +x docker/build-orbslam3 docker/build-helpers docker/orbslam3-entrypoint \
    && docker/build-orbslam3 \
    && docker/build-helpers \
    && test -x ${ORB_SLAM3_ROOT}/bin/sequence_observation_export \
    && test -x ${ORB_SLAM3_ROOT}/bin/rgbd_keyframes_to_ply \
    && test -x ${ORB_SLAM3_ROOT}/bin/scene_atlas_export \
    && ( ${ORB_SLAM3_ROOT}/bin/scene_atlas_export >/tmp/scene_atlas_export.usage 2>&1; test $? -eq 2 ) \
    && grep -q '^usage: scene_atlas_export ' /tmp/scene_atlas_export.usage \
    && ldconfig

ENV PATH="${ORB_SLAM3_ROOT}/bin:${ORB_SLAM3_ROOT}/Examples/RGB-D:${ORB_SLAM3_ROOT}/Examples/RGB-D-Inertial:${ORB_SLAM3_ROOT}/Examples/Stereo:${ORB_SLAM3_ROOT}/Examples/Monocular:${ORB_SLAM3_ROOT}/Examples/Monocular-Inertial:${ORB_SLAM3_ROOT}/Examples/Stereo-Inertial:${ORB_SLAM3_ROOT}/Examples/Calibration:${PATH}"

WORKDIR /work
ENTRYPOINT ["/opt/orbslam3/docker/orbslam3-entrypoint"]
CMD ["bash"]

// Scene-query ABI declarations derived from NVIDIA PhysX IPhysxSceneQuery.h:
// https://github.com/NVIDIA-Omniverse/PhysX/blob/3ca45ad36e97/ovphysx/ovruntime/include/omni/physx/IPhysxSceneQuery.h
// SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: BSD-3-Clause
// The Python loader restricts this adapter to the verified Kit 107.3.26 build
// and checks every startup ray against the public scalar API before arming.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <carb/Types.h>
#include <array>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
namespace py = pybind11;
namespace omni { namespace physx {
struct SceneQueryHitObject { uint64_t collision; uint64_t rigidBody; uint32_t protoIndex; };
struct SceneQueryHitLocation : SceneQueryHitObject {
    carb::Float3 normal; carb::Float3 position; float distance; uint32_t faceIndex; uint64_t material;
};
struct RaycastHit : SceneQueryHitLocation {};
// The only accessed entry is the first, unchanged raycastClosest function.
struct IPhysxSceneQuery {
    bool (*raycastClosest)(const carb::Float3&, const carb::Float3&, float, RaycastHit&, bool);
};
}}
static_assert(sizeof(omni::physx::RaycastHit) == 64, "Unsupported scene-query ABI");

class RayBatch {
    std::mutex mutex_, call_mutex_;
    std::condition_variable ready_, done_;
    std::vector<std::thread> workers_;
    std::function<void(int)> job_;
    unsigned generation_ = 0;
    int pending_ = 0, count_;
    bool stopping_ = false;
public:
    explicit RayBatch(int count) : count_(count) {
        if (count < 1 || count > 8) throw std::invalid_argument("ray threads must be in [1,8]");
        try {
            for (int id = 1; id < count_; ++id) workers_.emplace_back([this, id] {
                unsigned seen = 0;
                std::unique_lock<std::mutex> lock(mutex_);
                while (true) {
                    ready_.wait(lock, [&] { return stopping_ || generation_ != seen; });
                    if (stopping_) return;
                    seen = generation_;
                    auto job = job_;
                    lock.unlock();
                    job(id);
                    lock.lock();
                    if (--pending_ == 0) done_.notify_one();
                }
            });
        } catch (...) { close(); throw; }
    }
    void close() {
        std::lock_guard<std::mutex> call_lock(call_mutex_);
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        ready_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }
    ~RayBatch() { close(); }

    py::tuple scan(omni::physx::IPhysxSceneQuery& q,
            py::array_t<float, py::array::c_style | py::array::forcecast> origins,
            py::array_t<float, py::array::c_style | py::array::forcecast> directions) {
        auto o = origins.unchecked<2>(); auto d = directions.unchecked<2>();
        if (o.shape(1) != 3 || d.shape(1) != 3 || o.shape(0) == 0 || d.shape(0) == 0)
            throw std::invalid_argument("expected nonempty Nx3 arrays");
        auto n = o.shape(0) * d.shape(0);
        py::array_t<float> positions({n, py::ssize_t(3)});
        py::array_t<uint64_t> identities({n, py::ssize_t(2)});
        py::array_t<bool> hits(n);
        auto p = positions.mutable_unchecked<2>(); auto i = identities.mutable_unchecked<2>();
        auto h = hits.mutable_unchecked<1>();
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> call_lock(call_mutex_);
            if (stopping_) throw std::runtime_error("ray batch is closed");
            // Prime deferred query-tree updates before concurrent read calls.
            // No simulate/fetchResults or pose write occurs until all jobs finish.
            omni::physx::RaycastHit warm{};
            q.raycastClosest({o(0,0),o(0,1),o(0,2)}, {d(0,0),d(0,1),d(0,2)}, 4.5f, warm, false);
            auto run = [&](int id) {
                for (auto k = n*id/count_; k < n*(id+1)/count_; ++k) {
                    auto s = k/d.shape(0), r = k%d.shape(0);
                    omni::physx::RaycastHit v{};
                    h(k) = q.raycastClosest({o(s,0),o(s,1),o(s,2)},
                        {d(r,0),d(r,1),d(r,2)}, 4.5f, v, false);
                    i(k,0) = h(k) ? v.collision : 0; i(k,1) = h(k) ? v.rigidBody : 0;
                    p(k,0) = v.position.x; p(k,1) = v.position.y; p(k,2) = v.position.z;
                }
            };
            {
                std::lock_guard<std::mutex> lock(mutex_);
                job_ = run; pending_ = count_-1; ++generation_;
            }
            ready_.notify_all();
            run(0);
            std::unique_lock<std::mutex> lock(mutex_);
            done_.wait(lock, [&] { return pending_ == 0; });
            job_ = {};
        }
        // Intern identities in ray order, without sorting thousands of records.
        struct Hash {
            size_t operator()(const std::array<uint64_t,2>& key) const {
                return std::hash<uint64_t>{}(key[0]) ^ (std::hash<uint64_t>{}(key[1]) << 1);
            }
        };
        std::unordered_map<std::array<uint64_t,2>, uint32_t, Hash> table;
        std::vector<std::array<uint64_t,2>> keys{{0,0}};
        py::array_t<uint32_t> labels(n); auto l = labels.mutable_unchecked<1>();
        for (py::ssize_t k = 0; k < n; ++k) {
            if (!h(k)) { l(k) = 0; continue; }
            std::array<uint64_t,2> key{i(k,0),i(k,1)};
            auto found = table.find(key);
            if (found == table.end()) {
                auto id = keys.size(); keys.push_back(key); table.emplace(key,id); l(k) = id;
            } else l(k) = found->second;
        }
        py::array_t<uint64_t> unique({py::ssize_t(keys.size()),py::ssize_t(2)});
        auto u = unique.mutable_unchecked<2>();
        for (size_t k = 0; k < keys.size(); ++k) { u(k,0) = keys[k][0]; u(k,1) = keys[k][1]; }
        return py::make_tuple(hits,positions,identities,labels,unique);
    }
};
PYBIND11_MODULE(physx_ray_batch, m) {
    py::class_<RayBatch>(m,"RayBatch")
        .def(py::init<int>()).def("scan",&RayBatch::scan).def("close",&RayBatch::close);
}

// cs2-cv-aimbot.cpp
// Full-build: single file, no placeholders.
// g++ -std=c++20 -O3 cs2-cv-aimbot.cpp -lonnxruntime -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -levdev -o cs2-cv-aimbot
// sudo ./cs2-cv-aimbot

#include <linux/uinput.h>
#include <libevdev/libevdev.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>

static const char* MODEL   = "cs2head.onnx";   // 1-class model: 0=head
static const int   FOV     = 120;              // pixels radius
static const float SMOOTH   = 8.0f;
static const float TRIGGER  = 0.22f;           // crosshair overlap 22 % -> shoot

// ---------------- uinput mouse ----------------------------------
class UMouse {
    int fd;
public:
    UMouse() {
        fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
        if (fd < 0) throw std::runtime_error("uinput");
        uinput_setup usetup{};
        ui_set_evbit(fd, EV_REL);
        ui_set_evbit(fd, EV_KEY);
        ui_set_relbit(fd, REL_X);
        ui_set_relbit(fd, REL_Y);
        ui_set_keybit(fd, BTN_LEFT);
        std::strcpy(usetup.name, "cs2");
        usetup.id.bustype = BUS_USB;
        ioctl(fd, UI_DEV_SETUP, &usetup);
        ioctl(fd, UI_DEV_CREATE);
    }
    void move(int dx, int dy) {
        input_event ev[2] = {};
        ev[0] = {.type = EV_REL, .code = REL_X, .value = dx};
        ev[1] = {.type = EV_REL, .code = REL_Y, .value = dy};
        write(fd, ev, sizeof(ev));
        input_event sync = {.type = EV_SYN, .code = SYN_REPORT};
        write(fd, &sync, sizeof(sync));
    }
    void click() {
        input_event down = {.type = EV_KEY, .code = BTN_LEFT, .value = 1};
        input_event up   = {.type = EV_KEY, .code = BTN_LEFT, .value = 0};
        input_event sync = {.type = EV_SYN, .code = SYN_REPORT};
        write(fd, &down, sizeof(down));
        write(fd, &sync, sizeof(sync));
        write(fd, &up,   sizeof(up));
        write(fd, &sync, sizeof(sync));
    }
    ~UMouse() { ioctl(fd, UI_DEV_DESTROY); close(fd); }
};

// ---------------- evdev toggle key ------------------------------
class Toggle {
    libevdev *dev = nullptr;
    int fd = -1;
    std::atomic<bool> state{true};
public:
    Toggle() {
        fd = open("/dev/input/event3", O_RDONLY | O_NONBLOCK); // adjust to your keyboard
        if (fd < 0) throw std::runtime_error("keyboard");
        libevdev_new_from_fd(fd, &dev);
        std::thread([this]{
            input_event ev;
            while (libevdev_next_event(dev, LIBEVDEV_READ_FLAG_NORMAL, &ev) == 0 || libevdev_next_event(dev, LIBEVDEV_READ_FLAG_SYNC, &ev) == 0) {
                if (ev.type == EV_KEY && ev.code == KEY_F8 && ev.value == 1)
                    state = !state;
            }
        }).detach();
    }
    bool active() const { return state.load(); }
    ~Toggle() { if (dev) libevdev_free(dev); if (fd >= 0) close(fd); }
};

// ---------------- screen grab (DRM fallback to V4L2) ------------
cv::Mat grab() {
    static cv::VideoCapture cap(0, cv::CAP_V4L2);
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) throw std::runtime_error("grab failed");
    return frame;
}

// ---------------- head detector --------------------------------
std::vector<cv::Rect> detect(cv::Mat& img, Ort::Session& sess) {
    int64_t shape[] = {1,3,640,640};
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1/255.0, cv::Size(640,640), cv::Scalar(), true, false);
    Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(info, (float*)blob.data, blob.total(), shape, 4);
    auto out = sess.Run(Ort::RunOptions{nullptr}, "images", &tensor, 1, "output0", 1);
    float* d = out[0].GetTensorMutableData<float>();
    std::vector<cv::Rect> res;
    for (int i = 0; i < 8400; ++i) {
        float conf = d[4*8400 + i];
        if (conf < 0.35f) continue;
        float x = d[i], y = d[8400 + i], w = d[2*8400 + i], h = d[3*8400 + i];
        int x1 = int((x - w/2) * img.cols / 640);
        int y1 = int((y - h/2) * img.rows / 640);
        int x2 = int((x + w/2) * img.cols / 640);
        int y2 = int((y + h/2) * img.rows / 640);
        res.emplace_back(x1, y1, x2 - x1, y2 - y1);
    }
    return res;
}

// ---------------- main loop -------------------------------------
int main() {
    UMouse mouse;
    Toggle toggle;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cs2");
    Ort::SessionOptions opt;
    Ort::Session session(env, MODEL, opt);
    const int cross = 10; // crosshair size
    while (true) {
        if (!toggle.active()) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); continue; }
        cv::Mat frame = grab();
        int cx = frame.cols / 2, cy = frame.rows / 2;
        cv::circle(frame, {cx, cy}, FOV, {0,255,0}, 1);
        cv::line(frame, {cx - cross, cy}, {cx + cross, cy}, {0,255,0}, 1);
        cv::line(frame, {cx, cy - cross}, {cx, cy + cross}, {0,255,0}, 1);
        auto heads = detect(frame, session);
        int best_dx = 0, best_dy = 0;
        float best_area = 0;
        for (auto& r : heads) {
            int dx = (r.x + r.width / 2) - cx;
            int dy = (r.y + r.height / 2) - cy;
            float dist = std::sqrt(dx*dx + dy*dy);
            if (dist > FOV) continue;
            if (r.area() > best_area) { best_area = r.area(); best_dx = dx; best_dy = dy; }
            cv::rectangle(frame, r, {0,0,255}, 2);
        }
        if (best_area > 0) {
            mouse.move(best_dx / SMOOTH, best_dy / SMOOTH);
            cv::Rect crossbox(cx - 5, cy - 5, 10, 10);
            cv::Rect headbox(cx + best_dx - 5, cy + best_dy - 5, 10, 10);
            double overlap = (crossbox & headbox).area() / double(headbox.area());
            if (overlap > TRIGGER) mouse.click();
        }
        cv::imshow("cs2", frame);
        cv::waitKey(1);
    }
}

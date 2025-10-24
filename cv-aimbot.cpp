// cv-aimbot.cpp
// g++ -std=c++20 -O3 cv-aimbot.cpp -lonnxruntime -lopencv_core -lopencv_imgproc -lopencv_highgui -o cv-aimbot
// sudo ./cv-aimbot

#include <linux/uinput.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <thread>

static const char* MODEL = "yolov8n.onnx";

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
        std::strcpy(usetup.name, "aim");
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
    ~UMouse() { ioctl(fd, UI_DEV_DESTROY); close(fd); }
};

cv::Mat grabScreen() {
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) cap.open(0);
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) throw std::runtime_error("no screen");
    return frame;
}

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
        if (conf < 0.5f) continue;
        float x = d[i], y = d[8400 + i], w = d[2*8400 + i], h = d[3*8400 + i];
        int x1 = int((x - w/2) * img.cols / 640);
        int y1 = int((y - h/2) * img.rows / 640);
        int x2 = int((x + w/2) * img.cols / 640);
        int y2 = int((y + h/2) * img.rows / 640);
        res.emplace_back(x1, y1, x2 - x1, y2 - y1);
    }
    return res;
}

int main() {
    UMouse mouse;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "aim");
    Ort::SessionOptions opt;
    Ort::Session session(env, MODEL, opt);
    while (true) {
        cv::Mat frame = grabScreen();
        auto dets = detect(frame, session);
        int cx = frame.cols / 2, cy = frame.rows / 2;
        int best_dx = 0, best_dy = 0;
        float best = 1e9f;
        for (auto& r : dets) {
            int dx = (r.x + r.width / 2) - cx;
            int dy = (r.y + r.height / 2) - cy;
            float dist = std::sqrt(dx*dx + dy*dy);
            if (dist < best) { best = dist; best_dx = dx; best_dy = dy; }
        }
        if (best < 1e8f) mouse.move(best_dx / 8, best_dy / 8);
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
}

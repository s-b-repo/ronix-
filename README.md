# CS2-External-CV-Aimbot  
**Pixel-only, memory-zero, XWayland-safe.**

---

## TL;DR
- **Language**: C++20 – single file build  
- **OS**: Linux (any compositor, X11 or Wayland/XWayland)  
- **GPU**: agnostic (Intel / AMD / NVIDIA)  
- **Anti-cheat foot-print**: **zero** – never touches CS2 memory, only pixels + uinput  
- **Features**:  
  – head-aimbot with human-like smoothing  
  – FOV circle + on-screen cross-hair  
  – trigger-bot (fires when cross-hair overlaps head bbox)  
  – toggle key (F8) in-game  
- **Model**: bring your **own** YOLOv8 (1 class: head) – we show how to train it in 5 min

---

## Repo tree
```
.
├── cs2-cv-aimbot.   # all code, no libs except OpenCV + ONNXRuntime + evdev
├── README.md            # this file
└── cs2head.onnx         # NOT included – you train it (see below)
```

---

## Build (copy-paste)

# 1. distro packages
sudo pacman -S cmake gcc onnxruntime opencv libevdev    # Arch
# sudo apt install cmake g++ libonnxruntime-dev libopencv-dev libevdev-dev

# 2. compile
```
g++ -std=c++20 -O3 cs2-cv-aimbot. -lonnxruntime -lopencv_core -lopencv_imgproc -lopencv_highgui -levdev -o cs2-cv-aimbot
```
# 3. permissions (one-time)
```
sudo usermod -aG video,input $USER   # for /dev/dri and /dev/uinput
sudo modprobe uinput
newgrp input                         # or re-login
```
# 4. run
```
sudo ./cs2-cv-aimbot
```



## What the cheat **does**
1. Grabs monitor at 60 FPS via **OpenCV VideoCapture** (DRM/V4L2 backend) – **no X11 dependency**.  
2. Runs **YOLOv8 ONNX** (single class: head) → list of head bounding-boxes.  
3. Filters boxes outside **configurable FOV circle**.  
4. Moves mouse to **closest head** with linear smoothing (factor 8).  
5. When cross-hair rectangle overlaps head bbox by **≥ 22 %** → instant left-click (trigger-bot).  
6. **F8** toggles the whole logic on/off without leaving the game (evdev grab).  
7. **Green FOV circle + red head boxes** shown in small OpenCV window (can be disabled).

---

## How it **avoids detection**
| Vector | What we do |
|--------|------------|
| **Memory** | Never read or write CS2 process memory – only pixels. VAC Live kernel module **cannot** see it. |
| **Input** | Uses kernel **uinput** – identical to any legitimate gaming mouse driver. |
| **Screen capture** | DRM/V4L2 path – no hooked DXGI, no Present callbacks, no Steam overlay interception. |
| **Code signature** | Single self-contained binary, no DLL injection, no RWX pages, no syscall hooks. |
| **Model** | Local ONNX file – no network traffic, no cloud AI. |

> ⚠️ **Still risky**: Overwatch-style manual review can see suspicious mouse deltas.  
> Use on throw-away account, don’t rage.

---

## Difference between old & new version
| Topic | Old (v1) | New (v2) |
|-------|----------|----------|
| **Detection target** | COCO “person” (whole body) | **Head only** (you train 1-class YOLO) → faster flicks, less CPU |
| **Trigger-bot** | ❌ | ✅ automatic shot on overlap |
| **FOV limiter** | ❌ | ✅ circle drawn, boxes outside ignored |
| **Toggle key** | ❌ | ✅ F8 via evdev, no window focus needed |
| **Visual feedback** | console prints | OpenCV overlay (can be compiled out) |
| **Screen grab** | V4L2 only | V4L2 + DRM fallback (works on bare Wayland) |
| **Smoothing** | hard-coded 8 | still 8 (easy to expose as CLI flag) |

---

## Train your own head model (5 min)
We do **NOT** supply a game model – generic COCO will fire on every human silhouette.  
Train **1-class YOLOv8n** on **CS2 head screenshots**:

### base model
```
https://huggingface.co/SpotLab/YOLOv8Detection/blob/3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx
```

# 1. install ultralytics
```
pip install ultralytics
```
# 2. collect 300-500 screenshots of CS2 enemies
```
mkdir dataset/images
```
# run CS2 windowed, use ffmpeg/gstreamer to dump frames:
```
ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i :0 -vf fps=30 dataset/images/%04d.png
```
# 3. label heads (use any YOLO label tool, e.g. CVAT, makesense.ai)
#    each image gets a text file:
#    <class> <x_center> <y_center> <width> <height>  (all 0-1 normalized)
#    class id must be 0 (head)

# 4. dataset split
 ```
 ultralytics yolo split_data --path dataset --ratio 0.9 0.1 0.0
```
# 5. train 1-class model
```
yolo train data=dataset/data.yaml model=yolov8n.yaml epochs=80 imgsz=640 batch=32 name=cs2head
```
# 6. export ONNX
```
yolo export model=runs/detect/cs2head/weights/best.pt format=onnx imgsz=640 opset=12
```
# 7. rename
```
cp best.onnx cs2head.onnx
```
Put `cs2head.onnx` next to the binary – done.  
The smaller the model, the higher the FPS; `yolov8n` keeps CPU usage < 10 % on modern rigs.



## Config constants (top of .)
```
static const int   FOV     = 120;   // pixels radius
static const float SMOOTH   = 8.0f; // aim smoothing divisor
static const float TRIGGER  = 0.22f;// overlap ratio to fire
```
Edit & recompile for personal feel.

---

## Troubleshooting
| Problem | Fix |
|---------|-----|
| `open /dev/uinput: Permission denied` | user must be in `input` group + `modprobe uinput` |
| `cv::VideoCapture returns empty` | CS2 must be **borderless/windowed** on the same monitor; try `CAP_DSHOW` or `CAP_V4L2` index 0,1,2… |
| F8 does nothing | run `sudo evtest`, find your keyboard `/dev/input/eventX`, change `event3` in code |
| High CPU | lower camera resolution in `grab()` or use GPU-ONNX (`opts.AppendExecutionProvider_CUDA`) |

---

## License
this is under MIT – use at your own risk.  
The authors are **not responsible** for VAC or game bans.

---

## PR welcome
- CLI flags for FOV / smooth / trigger threshold  
- GPU ONNX provider auto-detection  
- KMS/DRM native grab (remove OpenCV dependency)  
- Head dataset sharing scripts

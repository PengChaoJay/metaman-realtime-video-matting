# Metaman Realtime Background Removal

一个面向数字人、直播和网页互动场景的实时视频背景去除服务。项目支持 RVM AI 人像背景去除、绿幕抠像、透明 RGBA Canvas、摄像头和本地视频输入，并提供 WebSocket 与 WebRTC 演示接口。

> 当前项目处于演示和技术验证阶段。生产环境使用前，请补充鉴权、限流、监控和 TURN 服务。

## 功能

- **RVM AI 背景去除**：使用 Robust Video Matting MobileNetV3，保留视频循环状态，减少快速动作时的边缘闪烁。
- **浏览器 GPU 加速**：TensorFlow.js 通过 WebGL 使用 Windows GPU，无需将 AI 模式的每一帧上传到服务端。
- **绿幕背景去除**：服务端实时色度键控，支持去绿溢色、运动残影抑制和强度调节。
- **透明 Canvas**：输出 RGBA 画面，可在当前页面内拖动数字人。
- **多档画质**：提供流畅 480p、高清 720p 和超清 1080p。
- **多种输入**：支持本地视频和浏览器摄像头。
- **实时状态**：RVM 模式显示 WebGL 后端和实际 FPS。
- **服务接口**：包含健康检查、WebSocket 逐帧处理和 WebRTC RGB/Alpha 打包示例。

## 工作方式

### RVM AI 模式

~~~text
本地视频或摄像头
        |
        v
TensorFlow.js + RVM MobileNetV3 INT8
        |
        v
Windows WebGL GPU
        |
        v
透明 RGBA Canvas
~~~

AI 模式直接在浏览器中推理，不经过 JPEG 上传和 PNG 回传，因此延迟更低。每个视频会话维护四组 RVM recurrent state，用于保持相邻帧的时序稳定性。

### 绿幕模式

~~~text
本地视频或摄像头
        |
        v
JPEG / WebSocket
        |
        v
Python 绿幕抠像 + 去绿溢色
        |
        v
透明 PNG / Canvas
~~~

### WebRTC 演示接口

普通 WebRTC 视频轨道没有通用的 Alpha 通道传输方式。服务端示例将 RGB 和 Alpha 横向打包进同一帧，浏览器再通过 WebGL Shader 合成为透明 Canvas，避免两路视频轨道不同步。

## 技术栈

- Python 3.12
- FastAPI
- OpenCV
- MediaPipe（服务端 WebRTC 回退模型）
- aiortc
- TensorFlow.js
- Robust Video Matting MobileNetV3
- WebGL / Canvas
- Docker

## 快速开始

### Docker

~~~bash
git clone https://github.com/<your-account>/metaman-realtime-background-removal.git
cd metaman-realtime-background-removal

docker build -t metaman-realtime-background-removal .
docker run --rm --name metaman-background-removal \
  -p 8090:8090 \
  metaman-realtime-background-removal
~~~

打开：

~~~text
http://localhost:8090/
~~~

页面中的 RVM AI 模式在浏览器端执行，因此普通页面体验只需映射端口 8090。如果需要使用服务端 /offer WebRTC 接口，Linux 环境建议使用 --network host，并根据部署网络配置 ICE/TURN。

### Python

要求 Python 3.10–3.12，推荐 Python 3.12。

Linux、macOS 或 WSL：

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-model-deps.txt
pip install --no-deps -r requirements-model.txt
uvicorn app.main:app --host 0.0.0.0 --port 8090
~~~

Windows PowerShell：

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-model-deps.txt
pip install --no-deps -r requirements-model.txt
uvicorn app.main:app --host 0.0.0.0 --port 8090
~~~

浏览器摄像头功能只能在 localhost 或 HTTPS 安全上下文中使用。

## 使用方法

1. 打开 http://localhost:8090/。
2. 选择“绿幕抠图”或“RVM AI 人像抠图”。
3. 选择画质档位。
4. 根据模式调节“去绿强度”或“边缘收紧”。
5. 点击“选择本地视频”，或者使用摄像头。
6. 在透明画面上按住鼠标或手指进行拖动。

RVM 模式运行后，状态栏会显示类似：

~~~text
RVM MobileNetV3 · WEBGL · 25.0 FPS
~~~

在 Edge 中可打开 edge://gpu，确认浏览器硬件加速状态。实际 FPS 取决于显卡、视频分辨率、浏览器和 RVM 内部下采样比例。

## API

### 健康检查

~~~http
GET /healthz
~~~

### WebSocket 逐帧处理

~~~text
WS /ws/process?mode=green&strength=0.78
WS /ws/process?mode=person&person_strength=0.65
~~~

客户端发送 JPEG 二进制帧，服务端返回透明 PNG。页面的 RVM AI 模式默认使用浏览器 WebGL，不调用 mode=person WebSocket。

### WebRTC Offer

~~~http
POST /offer
Content-Type: application/json

{
  "sdp": "...",
  "type": "offer"
}
~~~

返回 SDP answer。该接口主要用于演示 RGB 与 Alpha 单帧打包方案。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| PERSON_MODEL | mediapipe | 服务端 WebRTC 人像模型；页面 AI 模式使用浏览器 RVM |
| MASK_THRESHOLD | 动态 | MediaPipe 前景概率阈值 |
| MASK_FEATHER | 动态 | MediaPipe 边缘羽化范围 |
| MASK_TEMPORAL_SMOOTHING | 动态 | MediaPipe 时域平滑强度 |
| MASK_BLUR_SIZE | 3 | MediaPipe Alpha 模糊核大小 |
| MASK_EDGE_SHRINK | 动态 | MediaPipe 边缘内缩像素 |
| RVM_MODEL_PATH | /app/models/rvm_mobilenetv3_fp16.onnx | 可选服务端 ONNX 模型路径 |
| RVM_REQUIRE_CUDA | 1 | 可选服务端 RVM 是否必须使用 CUDA |
| RVM_ALLOW_FALLBACK | 1 | 服务端 RVM 失败时是否回退 MediaPipe |
| RVM_DOWNSAMPLE_RATIO | 自动 | 可选服务端 RVM 内部下采样比例 |
| LOG_LEVEL | INFO | 服务日志级别 |

## 项目结构

~~~text
app/
  main.py             FastAPI、WebSocket 和 WebRTC 接口
  remover.py          绿幕、MediaPipe 和可选 ONNX RVM 实现
  frame_processor.py  JPEG 到透明 PNG
  tracks.py           WebRTC RGB/Alpha 帧处理
static/
  index.html          演示页面和透明 Canvas
  rvm/                RVM TensorFlow.js INT8 模型
  vendor/             TensorFlow.js 浏览器运行库
tests/                单元测试和 WebRTC 往返测试
~~~

## 测试

~~~bash
pip install -r requirements-dev.txt
pip install -r requirements-model-deps.txt
pip install --no-deps -r requirements-model.txt
python -m unittest discover -s tests -v
~~~

## 已知限制

- 透明 Canvas 只能浮在当前网页内容上，不能跨浏览器标签页或覆盖其他桌面应用。跨应用透明置顶需要 Electron、Tauri 或原生窗口。
- RVM 的结果受原视频清晰度、运动模糊、人物占比和光照影响。
- 1080p 会增加 GPU、CPU 和内存占用；低性能设备建议使用 480p 或 720p。
- 服务端 WebRTC 接口尚未包含鉴权、会话配额、TURN 和多实例会话粘滞。
- GPL-3.0 对分发和衍生作品有相应义务；发布或商用前请自行评估合规要求。

## 第三方组件

- [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) — GPL-3.0
- [TensorFlow.js](https://github.com/tensorflow/tfjs) — Apache-2.0
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — Apache-2.0
- [FastAPI](https://github.com/fastapi/fastapi) — MIT
- [aiortc](https://github.com/aiortc/aiortc) — BSD-3-Clause
- [OpenCV](https://github.com/opencv/opencv) — Apache-2.0

第三方组件仍受各自许可证约束。

## License

本项目采用 **GNU General Public License v3.0 only（GPL-3.0-only）** 发布。完整协议见 [LICENSE](LICENSE)。

你可以使用、研究、修改和再分发本项目；当你分发本项目或其衍生作品时，需要遵守 GPL-3.0 的源代码开放、许可证保留等要求。本节仅为简要说明，不构成法律意见。

## 致谢

感谢 RVM、TensorFlow.js、MediaPipe、FastAPI、aiortc 和 OpenCV 社区提供的开源项目。

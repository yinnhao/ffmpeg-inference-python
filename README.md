# ffmpeg-inference-python:  image enhancement model inference workflow on video using ffmpeg-python

# Overview
当前图像/视频增强算法的官方代码往往不包含在视频文件上的推理，我们试图建立一个简单的框架，可以快速验证各种图像增强算法(包括超分、去噪、纹理色彩增强、超帧、sdr2hdr等)在视频上的效果。

ffmpeg具有强大的功能，但编写ffmpeg filter比较耗时，而ffmpeg-python使得我们可以经过简单改写算法代码，就可以完成视频推理。
# Dependencies
ffmpeg with x264/x265 (or other encoder you need) support 

Required Python packages:
numpy
ffmpeg-python
```
pip install ffmpeg-python
```
# Quickstart

继承video_infer创建新类，并重写单帧推理函数 def forward(self, x)， 注意x为ffmpeg的解除的yuv帧，尺寸为[3, h, w]

### demo 1
将彩色视频转为黑白视频
```python
from ffmpeg_infer import video_infer
class gray_video_infer(video_infer):
    def __init__(self, in_path, out_path, encode_params,  model=None, scale=1, in_pix_fmt="yuv444p", out_pix_fmt="yuv444p"):
        super(gray_video_infer, self).__init__(in_path, out_path, encode_params,  model, scale, in_pix_fmt, out_pix_fmt)

    def forward(self, x):
        y = x.copy()
        y[1, :, :] = 128
        y[2, :, :] = 128
        return y

in_path = "input.mp4"
out_path = "gray.mp4"
encode_params = ("libx264", "x264opts", "qp=12:bframes=3") # 输出视频编码参数
gray = gray_video_infer(in_path, out_path, encode_params, out_pix_fmt="yuv444p")
gray.infer()
del gray
``` 

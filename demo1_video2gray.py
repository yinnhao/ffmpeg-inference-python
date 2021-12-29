from ffmpeg_infer import video_infer

class gray_video_infer(video_infer):
    def __init__(self, in_path, out_path, encode_params,  model=None, scale=1, in_pix_fmt="yuv444p", out_pix_fmt="yuv444p"):
        super(gray_video_infer, self).__init__(in_path, out_path, encode_params,  model, scale, in_pix_fmt, out_pix_fmt)

    def forward(self, x):
        y = x.copy()
        y[1, :, :] = 128
        y[2, :, :] = 128
        return y


in_path = "/data/yh/video/gongxun_1_15s.mov"
out_path = "/data/yh/video/gongxun_1_15s_gray.mp4"
encode_params = ("libx264", "x264opts", "qp=12:bframes=3")
gray = gray_video_infer(in_path, out_path, encode_params, out_pix_fmt="yuv444p")
gray.infer()
# print(gray.get_video_info())
del gray
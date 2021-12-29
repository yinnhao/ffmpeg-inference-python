
import ffmpeg
import subprocess
import numpy as np

class video_infer:
    def __init__(self, file_name, save_name, encode_params, model=None, scale=1, in_pix_fmt="yuv444p", out_pix_fmt="yuv444p10le") -> None:
        self.file_name = file_name
        self.save_name = save_name
        self.model = model
        self.in_pix_fmt = in_pix_fmt
        self.out_pix_fmt = out_pix_fmt
        probe = ffmpeg.probe(self.file_name)
        self.video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        self.width = int(self.video_info['width'])
        self.height = int(self.video_info['height'])
        self.fps = round(float(self.video_info['r_frame_rate'].split('/')[0]) / float(self.video_info['r_frame_rate'].split('/')[1]))
        self.out_fps = self.fps
        self.out_width = self.width*scale
        self.out_height = self.height*scale
        self.encode_params = encode_params
        

    def get_video_info(self):
        return {"video name": self.file_name, "height":self.height, "width":self.width, "fps":self.fps}
    def start_ffmpeg_process1(self):
        args = (
            ffmpeg
                .input(self.file_name)
                .output('pipe:', color_primaries=9, color_trc=9, colorspace=9, format='rawvideo', pix_fmt=self.in_pix_fmt)
                .compile()
        )
        return subprocess.Popen(args, stdout=subprocess.PIPE)


    def start_ffmpeg_process2(self):
        args = (
            ffmpeg
                .input('pipe:', r=self.out_fps, color_primaries=9, color_trc=9, colorspace=9, format='rawvideo', pix_fmt=self.out_pix_fmt, s='{}x{}'.format(self.out_width, self.out_height))
                .output(self.save_name, **{'c:v': self.encode_params[0]}, **{self.encode_params[1]: self.encode_params[2]},  pix_fmt=self.out_pix_fmt)
                .overwrite_output()
                .compile()
        )
        return subprocess.Popen(args, stdin=subprocess.PIPE)


    def forward(self, *input):
        r"""Defines the computation performed at every call.

        Should be overridden by all subclasses.

        """
        raise NotImplementedError


    def read_frame(self):

        frame_size = self.height * self.width * 3 
        in_bytes = self.process1.stdout.read(frame_size)
        if len(in_bytes) == 0:
            self.frame = None
        else:
            assert len(in_bytes) == frame_size
            self.frame = (
                np
                    .frombuffer(in_bytes, np.uint8)
                    .reshape([3, self.height, self.width])
            )
        return self.frame
            
    def write_frame(self, frame):
        data_type_list = {"yuv420p":np.uint8, "yuv444p":np.uint8, "yuv420p10le":np.uint16, "yuv444p10le":np.uint16}
        data_type = data_type_list[self.out_pix_fmt]
        self.process2.stdin.write(
            frame
                .astype(data_type)
                .tobytes()
        )
    def infer(self):
        self.process1 = self.start_ffmpeg_process1()
        self.process2 = self.start_ffmpeg_process2()
        while(True):
            frame_in = self.read_frame()
            if frame_in is None:
                break
            frame_out = self.forward(frame_in)
            self.write_frame(frame_out)
        self.process1.wait()
        self.process2.stdin.close()
        self.process2.wait()


    def infer_multi_frames(self, N_in):
        self.process1 = self.start_ffmpeg_process1()
        self.process2 = self.start_ffmpeg_process2()
        batch = []
        frame_in = self.read_frame()
        batch.append(frame_in)
        for i in range(N_in//2):
            batch.append(frame_in)
        frame_index = 1
        while True:
            frame_index = frame_index + 1
            frame_in = self.read_frame()
            if frame_in is None:
                print('End of input stream')
                break
            batch.append(frame_in)
            if len(batch) == N_in:
                frame_out = self.forward(batch)
                self.write_frame(frame_out)
                batch.pop(0)
            
        for _ in range(N_in//2):
            batch.append(batch[-1])
            frame_out = self.forward(batch)
            self.write_frame(frame_out)
            batch.pop(0)
        self.process1.wait()
        self.process2.stdin.close()
        self.process2.wait()
   
    def __del__(self):
        print("distroy infer")





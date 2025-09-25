# Ocean Wave - 分钟分辨率、极值分段、潮速渐变、黄白蓝三色渐变
# 横轴为日期，纵轴为分钟，像素格子颜色按潮汐变化，动画按日期推进，颜色渐变，无边框
import pandas as pd
import numpy as np
from PIL import Image
import librosa
import os
import soundfile as sf



# 参数
W, H = 1280, 720  # 降低分辨率加速
FPS = 24
TARGET_SECONDS = 15
MINUTES_PER_DAY = 1440
MINUTES_PER_PIXEL = 10  # 每10分钟为一格
PIXELS_PER_DAY = MINUTES_PER_DAY // MINUTES_PER_PIXEL  # 144
NUM_WEEKS = (len(pd.read_csv('tide_hourly_2025.csv')['date']) + 6) // 7  # 向上取整


# 读取数据
hourly = pd.read_csv('tide_hourly_2025.csv')
extreme = pd.read_csv('tide_extreme_2025.csv')
dates = hourly['date'].tolist()
num_days = len(dates)

# 对每一天插值生成每分钟潮高
def interpolate_day(row):
    hvals = [float(getattr(row, f'h{i}')) for i in range(24)]
    hvals.append(hvals[0])
    x = np.arange(0, 25)*60
    return np.interp(np.arange(1440), x, hvals)

# 生成所有天的分钟潮高矩阵
minute_matrix = np.zeros((MINUTES_PER_DAY, num_days))
for i, row in enumerate(hourly.itertuples()):
    minute_matrix[:, i] = interpolate_day(row)

# 降采样为每5分钟一格
pixel_matrix = np.zeros((PIXELS_PER_DAY, num_days))
for i in range(num_days):
    for p in range(PIXELS_PER_DAY):
        pixel_matrix[p, i] = np.mean(minute_matrix[p*MINUTES_PER_PIXEL:(p+1)*MINUTES_PER_PIXEL, i])


# 计算每周的潮高（每5分钟平均）
weekly_pixel_matrix = np.zeros((PIXELS_PER_DAY, NUM_WEEKS))
for week in range(NUM_WEEKS):
    vals_by_pixel = [[] for _ in range(PIXELS_PER_DAY)]
    for day_idx in range(week*7, min((week+1)*7, num_days)):
        for p in range(PIXELS_PER_DAY):
            vals_by_pixel[p].append(pixel_matrix[p, day_idx])
    for p in range(PIXELS_PER_DAY):
        if vals_by_pixel[p]:
            weekly_pixel_matrix[p, week] = np.mean(vals_by_pixel[p])
        else:
            weekly_pixel_matrix[p, week] = np.nan

def make_pixel_matrix():
    # 新像素矩阵：纵轴288（每5分钟），横轴为周数
    mat = np.ones((PIXELS_PER_DAY, NUM_WEEKS, 3), dtype=np.uint8) * 255
    # 归一化潮高（全局最大最小）
    vmin = np.nanmin(weekly_pixel_matrix)
    vmax = np.nanmax(weekly_pixel_matrix)
    for x in range(NUM_WEEKS):
        for y in range(PIXELS_PER_DAY):
            h = weekly_pixel_matrix[y, x]
            if np.isnan(h):
                continue
            norm = (h - vmin) / (vmax - vmin + 1e-6)
            # 颜色映射：低为宝蓝(0,60,255)，高为橙红(255,100,0)，中间渐变
            r = int(0   * (1-norm) + 255 * norm)
            g = int(60  * (1-norm) + 100 * norm)
            b = int(255 * (1-norm) + 0   * norm)
            mat[y, x] = (r, g, b)
    return mat


# 用鼓点（强拍）驱动动画
def get_downbeat_times(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times

def make_frames(audio_path, music_duration, downbeat_times):
    frames = []
    mat = make_pixel_matrix()  # shape: (PIXELS_PER_DAY, NUM_WEEKS, 3)
    total_pixels = NUM_WEEKS * PIXELS_PER_DAY
    # 前奏阶段每个鼓点推进50个像素点，主旋律阶段均分
    PRELUDE_SECONDS = 15  # 前奏时长（秒），可根据实际音乐调整
    PIXELS_PER_PRELUDE_BEAT = 50
    pixels_per_beat_main = max(1, (total_pixels - PIXELS_PER_PRELUDE_BEAT * sum(downbeat_times < PRELUDE_SECONDS)) // max(1, len(downbeat_times) - sum(downbeat_times < PRELUDE_SECONDS)))
    # 随机推进
    all_indices = [(y, x) for x in range(NUM_WEEKS) for y in range(PIXELS_PER_DAY)]
    np.random.seed(42)
    unused_indices = set(all_indices)
    frame_mat = np.ones_like(mat) * 255
    frame_times = list(downbeat_times) + [music_duration]
    for i in range(len(downbeat_times)):
        # 判断是否为前奏
        if downbeat_times[i] < PRELUDE_SECONDS:
            n_new = min(len(unused_indices), PIXELS_PER_PRELUDE_BEAT)
        else:
            n_new = min(len(unused_indices), pixels_per_beat_main)
        # 随机抽取未出现过的像素
        if n_new > 0:
            new_indices = np.random.choice(len(unused_indices), n_new, replace=False)
            unused_indices_list = list(unused_indices)
            for idx in new_indices:
                y, x = unused_indices_list[idx]
                frame_mat[y, x, :] = mat[y, x, :]
                unused_indices.remove((y, x))
        raw_img = Image.fromarray(frame_mat.astype(np.uint8), mode='RGB')
        img = raw_img.resize((W, H), resample=Image.NEAREST)
        start = frame_times[i]
        end = min(frame_times[i+1], music_duration)  # 不超过音频时长
        n_frames = max(1, int(np.round((end - start) * FPS)))
        for fidx in range(n_frames):
            if (len(frames) / FPS) >= music_duration:
                break
            frames.append(np.array(img))
            if (len(frames) % 50) == 0:
                print(f"已生成帧数: {len(frames)} / 预计总帧数约{int(music_duration*FPS)}")
        if (len(frames) / FPS) >= music_duration:
            break
    # 最后一帧为全部像素点都已出现的完整画面
    full_img = Image.fromarray(mat.astype(np.uint8), mode='RGB').resize((W, H), resample=Image.NEAREST)
    full_frame = np.array(full_img)
    target_frames = int(music_duration * FPS)
    if len(frames) > target_frames:
        frames = frames[:target_frames-1]
        frames.append(full_frame)
    elif len(frames) < target_frames and frames:
        frames.extend([full_frame] * (target_frames - len(frames)))
    else:
        if frames:
            frames[-1] = full_frame

    # 0:28秒起右上角加字“Ocean Save”
    from PIL import ImageDraw, ImageFont
    text = "Ocean Save"
    font_size = int(H * 0.08)
    try:
        font = ImageFont.truetype("led_counter-7.ttf", font_size)
    except:
        font = ImageFont.load_default()
    text_color = (255,255,255)
    text_margin = 10
    start_frame = int(27 * FPS)
    for i in range(start_frame, len(frames)):
        img = Image.fromarray(frames[i])
        draw = ImageDraw.Draw(img)
        # Pillow >=8.0.0 推荐用textbbox获取宽高
        bbox = draw.textbbox((0,0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((W-w-text_margin, text_margin), text, font=font, fill=text_color)
        frames[i] = np.array(img)
    return frames



# 自动将mp3转为wav，优先用wav合成音轨，确保兼容性
mp3_path = "OceanWave-DecaJoins.mp3"
wav_path = "OceanWave-DecaJoins_temp.wav"
if not os.path.exists(wav_path):
    try:
        y, sr = librosa.load(mp3_path, sr=None)
        sf.write(wav_path, y, sr)
        print(f"已自动将mp3转为wav: {wav_path}")
    except Exception as e:
        print(f"mp3转wav失败: {e}")

# 用鼓点（强拍）驱动动画，视频时长严格等于音乐时长
downbeat_times = get_downbeat_times(mp3_path)

# 生成帧图片序列

music_duration = librosa.get_duration(filename=wav_path if os.path.exists(wav_path) else mp3_path)
frames = make_frames(mp3_path, music_duration, downbeat_times)
print(f"生成帧数: {len(frames)} (与音乐时长和鼓点同步)")
if frames:
    print(f"首帧shape: {frames[0].shape}")


# 保存帧为图片序列
output_dir = 'frames_output'
os.makedirs(output_dir, exist_ok=True)
for idx, frame in enumerate(frames):
    img = Image.fromarray(frame.astype(np.uint8), mode='RGB')
    img.save(os.path.join(output_dir, f'frame_{idx+1:05d}.png'))
print(f'已保存 {len(frames)} 帧到 {output_dir}/')

# 自动用ffmpeg合成mp4视频
import subprocess
audio_file = wav_path if os.path.exists(wav_path) else mp3_path
output_mp4 = 'tide_pixel_art.mp4'
ffmpeg_cmd = [
    'ffmpeg', '-y',
    '-framerate', str(FPS),
    '-i', f'{output_dir}/frame_%05d.png',
    '-i', audio_file,
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-c:a', 'aac',
    '-shortest',
    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
    output_mp4
]
try:
    print('正在用ffmpeg合成mp4视频...')
    subprocess.run(ffmpeg_cmd, check=True)
    print(f'视频已生成：{output_mp4}')
    
    # 清理临时文件
    if os.path.exists(wav_path):
        os.remove(wav_path)
        print(f'已清理临时文件：{wav_path}')
except Exception as e:
    print(f'ffmpeg合成视频失败: {e}')

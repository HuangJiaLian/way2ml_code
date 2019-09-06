'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-09-06 19:27:17
@LastEditors: Jack Huang
@LastEditTime: 2019-09-06 19:38:15
'''
# 得到列表
import glob
videos = glob.glob("./*.mp4")
videos.sort()
print(videos)
# print()
from moviepy.editor import *
import moviepy.editor as mpe

clips = [VideoFileClip(video) for video in videos]
concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile('output.mp4')
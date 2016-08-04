
# Save the files as png-files with the correct resolution


# Standard resolutions:
# 720 HD = 1280x720
# 1080 HD = 1920x1080

# Build the mpeg_file
ffmpeg -framerate 60 -i output.%04d.png -s:v 1920x1080 -c:v libx264 -crf 20 -pix_fmt yuv420p movie.mp4

# note the option -pix_fmt yuv420p which is necessary to play the video via QuickTime
# removing the png files
# rm output.*.png

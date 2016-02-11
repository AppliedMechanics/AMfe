
# Standard resolutions:
# 720 HD = 1280x720
# 1080 HD = 1920x1080

# Build the mpeg_file
ffmpeg -framerate 20 -i output.%04d.png -s:v 1920x1080 -c:v libx264 -crf 20 movie.mp4

# removing the png files
# rm output.*.png

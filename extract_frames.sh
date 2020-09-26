starttime=`date +%s`
videos_list=(sample_01)
for videoname in "${videos_list[@]}"; do
    save_path="images_root/${videoname}"
    mkdir -p $save_path
    ffmpeg -i "input_videos/${videoname}.mp4" -q:v 2 "${save_path}/%05d.jpg" -hide_banner
done
endtime=`date +%s`

echo `expr $endtime - $starttime`
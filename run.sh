# %%shell # google colab

# get cam
start_video=1
end_video=25

videos_list=()

# create_video list
for i in $(seq $start_video $end_video); do
  videoname=""
  if ((i < 10)); then
    videoname="cam_0"$i
  else
    videoname="cam_"$i
  fi
  videos_list+=($videoname)
done


echo "${videos_list[@]}"
input_videos="data/test_data"
mkdir -p $input_videos

# Extract frames
starttime=`date +%s`
images_root="images_root"
for videoname in "${videos_list[@]}"; do
  save_path="${images_root}/${videoname}"
  mkdir -p $save_path
  ffmpeg -i "${input_videos}/${videoname}.mp4" -q:v 2 "${save_path}/%05d.jpg" -hide_banner
done
endtime=`date +%s`
echo "Extract time "`expr $endtime - $starttime` "s"

# Run
tracking_outputs="track_outputs"
counting_outputs='counting_outputs'

mkdir -p $tracking_outputs
mkdir -p $counting_outputs

weight="detection_model/effdet-d5-640-second.pt"
conf_thres=0.4
iou_thres=0.5
max_age=1
thresh1=0.5
thresh2=0.9

for videoname in "${videos_list[@]}"; do
    # echo "Running " $videoname        
    python main.py --videoname ${videoname} --detecting --tracking --counting --count-outputs-root ${counting_outputs} --weight ${weight} --model-type "effdet5" --image-size-effdet 640 --conf-thres ${conf_thres} --track-features-type "label" --iou-thres ${iou_thres} --max-age ${max_age} --tracker-thresh1 ${thresh1} --tracker-thresh2 ${thresh2} --tracker-thresh3 ${thresh1}
done

submission_output="data/submission_output"
mkdir -p $submission_output
python merge_counting_outputs.py --root ${counting_outputs} --output ${submission_output} 

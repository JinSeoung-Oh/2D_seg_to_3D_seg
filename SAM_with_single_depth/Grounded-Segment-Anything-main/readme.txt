Follow Grounded-Segment-Anything github

https://github.com/IDEA-Research/Grounded-Segment-Anything

Just add grounded_sam_demo_get_coordination.py in Grounded-Segment-Anything-main file
I cannot upload my full code because of my personal problem.

After that, enter below command in your kernel

export CUDA_VISIBLE_DEVICES=0
python grounded_sam_demo_get_coordination.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo1.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "car" \
  --device "cuda"
  
  
Then you can get json file like this:

        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
            'mask' : mask.cpu().numpy().tolist(),
            'mask_coor': coor,
            'color' : color
        })

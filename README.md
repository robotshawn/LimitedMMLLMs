# LimitedMMLLMs
有限场景的多模态“大”模型，争取做到，“麻雀虽小，五脏俱全”

以下目录~/workspace/vlm_limited是本地的，大家可以将其替换为.../LimitedMMLLMs。基本代码都在脚本里面，数据可以自己按照下面的流程构建。

硬件条件：
Intel® Core™ i7-7820X CPU @ 3.60GHz × 16，Ubuntu 22.04.4 LTS，nvidia rtx 5070 Ti，现有环境：
fight@fight-System-Product-Name:~$ nvidia-smi
Tue Feb 17 20:40:37 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070 Ti     Off |   00000000:65:00.0  On |                  N/A |
|  0%   42C    P8             22W /  300W |     692MiB /  16303MiB |     20%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3023      G   /usr/lib/xorg/Xorg                      314MiB |
|    0   N/A  N/A            3430      G   /usr/bin/gnome-shell                     57MiB |
|    0   N/A  N/A            4788      G   .../7766/usr/lib/firefox/firefox        211MiB |
+-----------------------------------------------------------------------------------------+
fight@fight-System-Product-Name:~$ source ~/venvs/isaacsim511/bin/activate
(isaacsim511) fight@fight-System-Product-Name:~$ python
Python 3.11.0rc1 (main, Aug 12 2022, 10:02:14) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> 

数据选择：
mkdir -p *** && cd ***
wget -c https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip
...
cd ~/workspace/vlm_limited/data
mkdir -p clevr_raw
unzip -q CLEVR_v1.0.zip -d clevr_raw
议用 TRL 推荐的 vision conversational 格式（更通用、对 VLM 更友好）：

每条样本大概长这样（写入 jsonl 文件）：
{
  "image": "ABSOLUTE_OR_REL_PATH.png",
  "messages": [
    {"role":"user","content":[{"type":"image"},{"type":"text","text":"What color is the sphere?"}]},
    {"role":"assistant","content":[{"type":"text","text":"red"}]}
  ]
}
CLEVR 本身就是“受限场景”：固定对象集合（形状/颜色/材质/大小/空间关系）+ 模板化问题，非常适合作为你的小 VLM 的主训练集。
在你的工程里建个 scripts 目录：
cd ~/workspace/vlm_limited
mkdir -p scripts data/processed
python scripts/prepare_clevr_vlm.py \
  --clevr_root ~/workspace/vlm_limited/data/clevr_raw/CLEVR_v1.0 \
  --split train \
  --out  ~/workspace/vlm_limited/data/processed/clevr_train_20k.jsonl \
  --max_samples 20000

python scripts/prepare_clevr_vlm.py \
  --clevr_root ~/workspace/vlm_limited/data/clevr_raw/CLEVR_v1.0 \
  --split val \
  --out  ~/workspace/vlm_limited/data/processed/clevr_val_2k.jsonl \
  --max_samples 2000



(vlm) fight@fight-System-Product-Name:~/workspace/vlm_limited$ ALL_PROXY= all_proxy= CUDA_VISIBLE_DEVICES=0 python scripts/train_llava_clevr_qlora.py
The image processor of type `CLIPImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Fetching 3 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 20460.02it/s]
Download complete: : 0.00B [00:00, ?B/s]                                                                                                                                                                                                                                                                                                                      | 0/3 [00:00<?, ?it/s]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 686/686 [00:04<00:00, 148.85it/s, Materializing param=model.vision_tower.vision_model.pre_layrnorm.weight]
trainable params: 42,336,256 || all params: 7,105,763,328 || trainable%: 0.5958
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 2, 'bos_token_id': 1}.
  0%|                                                                                                                                                                                                                                                                                                                                                      | 0/1250 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
{'loss': '7.181', 'grad_norm': '0.04198', 'learning_rate': '9.848e-05', 'epoch': '0.016'}                                                                                                                                                                                                                                                                                           
{'loss': '4.239', 'grad_norm': '0.0196', 'learning_rate': '9.688e-05', 'epoch': '0.032'}                                                                                                                                                                                                                                                                                                    
{'loss': '4.086', 'grad_norm': '0.01518', 'learning_rate': '9.528e-05', 'epoch': '0.048'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.903', 'grad_norm': '0.01274', 'learning_rate': '9.368e-05', 'epoch': '0.064'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.772', 'grad_norm': '0.007325', 'learning_rate': '9.208e-05', 'epoch': '0.08'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.73', 'grad_norm': '0.005145', 'learning_rate': '9.048e-05', 'epoch': '0.096'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.719', 'grad_norm': '0.005996', 'learning_rate': '8.888e-05', 'epoch': '0.112'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.712', 'grad_norm': '0.00567', 'learning_rate': '8.728e-05', 'epoch': '0.128'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.711', 'grad_norm': '0.006999', 'learning_rate': '8.568e-05', 'epoch': '0.144'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.709', 'grad_norm': '0.007719', 'learning_rate': '8.408e-05', 'epoch': '0.16'}                                                                                                                                                                                                                                                                                                   
{'eval_loss': '3.707', 'eval_runtime': '512.7', 'eval_samples_per_second': '3.901', 'eval_steps_per_second': '3.901', 'epoch': '0.16'}                                                                                                                                                                                                                                                      
{'loss': '3.704', 'grad_norm': '0.006183', 'learning_rate': '8.248e-05', 'epoch': '0.176'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.706', 'grad_norm': '0.006485', 'learning_rate': '8.088e-05', 'epoch': '0.192'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.702', 'grad_norm': '0.0056', 'learning_rate': '7.928e-05', 'epoch': '0.208'}                                                                                                                                                                                                                                                                                                    
{'loss': '3.703', 'grad_norm': '0.006525', 'learning_rate': '7.768e-05', 'epoch': '0.224'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.706', 'grad_norm': '0.004972', 'learning_rate': '7.608e-05', 'epoch': '0.24'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.702', 'grad_norm': '0.006654', 'learning_rate': '7.448e-05', 'epoch': '0.256'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.703', 'grad_norm': '0.00627', 'learning_rate': '7.288e-05', 'epoch': '0.272'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.707', 'grad_norm': '0.006467', 'learning_rate': '7.128e-05', 'epoch': '0.288'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.701', 'grad_norm': '0.006338', 'learning_rate': '6.968e-05', 'epoch': '0.304'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.702', 'grad_norm': '0.005495', 'learning_rate': '6.808e-05', 'epoch': '0.32'}                                                                                                                                                                                                                                                                                                   
{'eval_loss': '3.705', 'eval_runtime': '513.9', 'eval_samples_per_second': '3.892', 'eval_steps_per_second': '3.892', 'epoch': '0.32'}                                                                                                                                                                                                                                                      
{'loss': '3.705', 'grad_norm': '0.006272', 'learning_rate': '6.648e-05', 'epoch': '0.336'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.703', 'grad_norm': '0.005249', 'learning_rate': '6.488e-05', 'epoch': '0.352'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.701', 'grad_norm': '0.005648', 'learning_rate': '6.328e-05', 'epoch': '0.368'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.703', 'grad_norm': '0.006779', 'learning_rate': '6.168e-05', 'epoch': '0.384'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.703', 'grad_norm': '0.005384', 'learning_rate': '6.008e-05', 'epoch': '0.4'}                                                                                                                                                                                                                                                                                                    
{'loss': '3.697', 'grad_norm': '0.005296', 'learning_rate': '5.848e-05', 'epoch': '0.416'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.704', 'grad_norm': '0.005292', 'learning_rate': '5.688e-05', 'epoch': '0.432'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.704', 'grad_norm': '0.005577', 'learning_rate': '5.528e-05', 'epoch': '0.448'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.706', 'grad_norm': '0.005483', 'learning_rate': '5.368e-05', 'epoch': '0.464'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.709', 'grad_norm': '0.006334', 'learning_rate': '5.208e-05', 'epoch': '0.48'}                                                                                                                                                                                                                                                                                                   
{'eval_loss': '3.704', 'eval_runtime': '512.9', 'eval_samples_per_second': '3.899', 'eval_steps_per_second': '3.899', 'epoch': '0.48'}                                                                                                                                                                                                                                                      
{'loss': '3.701', 'grad_norm': '0.005731', 'learning_rate': '5.048e-05', 'epoch': '0.496'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.705', 'grad_norm': '0.005876', 'learning_rate': '4.888e-05', 'epoch': '0.512'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.705', 'grad_norm': '0.005383', 'learning_rate': '4.728e-05', 'epoch': '0.528'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.701', 'grad_norm': '0.006009', 'learning_rate': '4.568e-05', 'epoch': '0.544'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.707', 'grad_norm': '0.004838', 'learning_rate': '4.408e-05', 'epoch': '0.56'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.702', 'grad_norm': '0.005468', 'learning_rate': '4.248e-05', 'epoch': '0.576'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.7', 'grad_norm': '0.005584', 'learning_rate': '4.088e-05', 'epoch': '0.592'}                                                                                                                                                                                                                                                                                                    
{'loss': '3.7', 'grad_norm': '0.005224', 'learning_rate': '3.928e-05', 'epoch': '0.608'}                                                                                                                                                                                                                                                                                                    
{'loss': '3.704', 'grad_norm': '0.006067', 'learning_rate': '3.768e-05', 'epoch': '0.624'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.705', 'grad_norm': '0.004949', 'learning_rate': '3.608e-05', 'epoch': '0.64'}                                                                                                                                                                                                                                                                                                   
{'eval_loss': '3.704', 'eval_runtime': '513.4', 'eval_samples_per_second': '3.896', 'eval_steps_per_second': '3.896', 'epoch': '0.64'}                                                                                                                                                                                                                                                      
{'loss': '3.704', 'grad_norm': '0.005313', 'learning_rate': '3.448e-05', 'epoch': '0.656'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.703', 'grad_norm': '0.005788', 'learning_rate': '3.288e-05', 'epoch': '0.672'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.704', 'grad_norm': '0.005156', 'learning_rate': '3.128e-05', 'epoch': '0.688'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.706', 'grad_norm': '0.005283', 'learning_rate': '2.968e-05', 'epoch': '0.704'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.702', 'grad_norm': '0.005408', 'learning_rate': '2.808e-05', 'epoch': '0.72'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.703', 'grad_norm': '0.005762', 'learning_rate': '2.648e-05', 'epoch': '0.736'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.702', 'grad_norm': '0.006458', 'learning_rate': '2.488e-05', 'epoch': '0.752'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.699', 'grad_norm': '0.00486', 'learning_rate': '2.328e-05', 'epoch': '0.768'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.703', 'grad_norm': '0.005526', 'learning_rate': '2.168e-05', 'epoch': '0.784'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.707', 'grad_norm': '0.006351', 'learning_rate': '2.008e-05', 'epoch': '0.8'}                                                                                                                                                                                                                                                                                                    
{'eval_loss': '3.705', 'eval_runtime': '514.5', 'eval_samples_per_second': '3.887', 'eval_steps_per_second': '3.887', 'epoch': '0.8'}                                                                                                                                                                                                                                                       
{'loss': '3.704', 'grad_norm': '0.00646', 'learning_rate': '1.848e-05', 'epoch': '0.816'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.703', 'grad_norm': '0.00556', 'learning_rate': '1.688e-05', 'epoch': '0.832'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.699', 'grad_norm': '0.006921', 'learning_rate': '1.528e-05', 'epoch': '0.848'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.7', 'grad_norm': '0.005444', 'learning_rate': '1.368e-05', 'epoch': '0.864'}                                                                                                                                                                                                                                                                                                    
{'loss': '3.706', 'grad_norm': '0.004941', 'learning_rate': '1.208e-05', 'epoch': '0.88'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.703', 'grad_norm': '0.006612', 'learning_rate': '1.048e-05', 'epoch': '0.896'}                                                                                                                                                                                                                                                                                                  
{'loss': '3.704', 'grad_norm': '0.005449', 'learning_rate': '8.88e-06', 'epoch': '0.912'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.706', 'grad_norm': '0.006119', 'learning_rate': '7.28e-06', 'epoch': '0.928'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.703', 'grad_norm': '0.005744', 'learning_rate': '5.68e-06', 'epoch': '0.944'}                                                                                                                                                                                                                                                                                                   
{'loss': '3.702', 'grad_norm': '0.005323', 'learning_rate': '4.08e-06', 'epoch': '0.96'}                                                                                                                                                                                                                                                                                                    
{'eval_loss': '3.705', 'eval_runtime': '513.4', 'eval_samples_per_second': '3.896', 'eval_steps_per_second': '3.896', 'epoch': '0.96'}                                                                                                                                                                                                                                                      
{'loss': '3.704', 'grad_norm': '0.00697', 'learning_rate': '2.48e-06', 'epoch': '0.976'}                                                                                                                                                                                                                                                                                                    
{'loss': '3.704', 'grad_norm': '0.006008', 'learning_rate': '8.8e-07', 'epoch': '0.992'}                                                                                                                                                                                                                                                                                                    
{'train_runtime': '2.035e+04', 'train_samples_per_second': '0.983', 'train_steps_per_second': '0.061', 'train_loss': '3.779', 'epoch': '1'}                                                                                                                                                                                                                                                 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1250/1250 [5:39:07<00:00, 16.28s/it]
[OK] Saved to: out/llava_clevr_qlora

(vlm) fight@fight-System-Product-Name:~/workspace/vlm_limited$ ALL_PROXY= all_proxy= CUDA_VISIBLE_DEVICES=0 python scripts/infer_llava_lora.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Fetching 3 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 10556.13it/s]
Download complete: : 0.00B [00:00, ?B/s]                                                                                                                                                                                                                                                                                                                      | 0/3 [00:00<?, ?it/s]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 686/686 [00:04<00:00, 145.72it/s, Materializing param=model.vision_tower.vision_model.pre_layrnorm.weight]
================================================================================
[1] Q : What number of things are either small blue things right of the large yellow matte thing or small shiny objects that are to the right of the blue rubber ball?
     GT: 4
     PR: 3
     OK: False
================================================================================
[2] Q : Is the size of the brown block that is on the left side of the large brown matte object the same as the rubber cylinder on the right side of the large metal sphere?
     GT: yes
     PR: no
     OK: False
================================================================================
[3] Q : What number of other green objects have the same material as the big green thing?
     GT: 0
     PR: 0
     OK: True
================================================================================
[4] Q : Are there fewer large objects that are to the right of the small metallic sphere than cylinders in front of the red rubber sphere?
     GT: yes
     PR: no
     OK: False
================================================================================
[5] Q : How many other small objects are the same shape as the tiny gray metallic object?
     GT: 1
     PR: 0
     OK: False
================================================================================
[6] Q : What is the shape of the cyan rubber thing that is the same size as the brown shiny cylinder?
     GT: sphere
     PR: sphere
     OK: True
================================================================================
[7] Q : The cyan rubber object that is the same size as the purple object is what shape?
     GT: cube
     PR: cube
     OK: True
================================================================================
[8] Q : What is the color of the shiny object that is on the left side of the tiny matte thing and behind the cyan matte object?
     GT: blue
     PR: red
     OK: False
================================================================================
[9] Q : There is a large object that is the same material as the small cyan thing; what is its shape?
     GT: cylinder
     PR: sphere
     OK: False
================================================================================
[10] Q : Is there any other thing that has the same size as the green sphere?
     GT: yes
     PR: yes
     OK: True
================================================================================
[11] Q : There is a block that is both behind the tiny brown metal cube and in front of the cyan thing; how big is it?
     GT: small
     PR: small
     OK: True
================================================================================
[12] Q : How many red cubes have the same material as the tiny purple thing?
     GT: 1
     PR: 0
     OK: False
================================================================================
[13] Q : How many objects are big gray rubber cylinders or small yellow metallic things?
     GT: 1
     PR: 1
     OK: True
================================================================================
[14] Q : Are the big brown cube and the big blue cylinder left of the big purple block made of the same material?
     GT: no
     PR: yes
     OK: False
================================================================================
[15] Q : How many red cubes are the same size as the blue matte thing?
     GT: 1
     PR: 0
     OK: False
================================================================================
[16] Q : Is the size of the blue rubber object in front of the green rubber sphere the same as the yellow block behind the large purple metal ball?
     GT: yes
     PR: no
     OK: False
================================================================================
[17] Q : Is there a cylinder of the same color as the tiny cube?
     GT: yes
     PR: no
     OK: False
================================================================================
[18] Q : Do the rubber ball and the shiny block have the same color?
     GT: no
     PR: no
     OK: True
================================================================================
[19] Q : Is there another rubber thing of the same shape as the big purple rubber thing?
     GT: yes
     PR: no
     OK: False
================================================================================
[20] Q : Is the shape of the red thing that is left of the big purple metal block the same as the shiny object that is on the right side of the purple shiny cube?
     GT: yes
     PR: no
     OK: False
================================================================================
Sample accuracy: 7/20 = 0.350
(vlm) fight@fight-System-Product-Name:~/workspace/vlm_limited$ ALL_PROXY= all_proxy= CUDA_VISIBLE_DEVICES=0 python scripts/eval_clevr_exact_match.py
Fetching 3 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 12813.56it/s]
Download complete: : 0.00B [00:00, ?B/s]                                                                                                                                                                                                                                                                                                                      | 0/3 [00:00<?, ?it/s]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 686/686 [00:04<00:00, 142.32it/s, Materializing param=model.vision_tower.vision_model.pre_layrnorm.weight]
200  acc=0.570
400  acc=0.570
600  acc=0.538
800  acc=0.551
1000  acc=0.552
1200  acc=0.537
1400  acc=0.531
1600  acc=0.534
1800  acc=0.536
2000  acc=0.537
Final: 1075/2000  acc=0.5375


(vlm) fight@fight-System-Product-Name:~/workspace/vlm_limited$ ALL_PROXY= all_proxy= CUDA_VISIBLE_DEVICES=0 python scripts/eval_clevr_breakdown.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Fetching 3 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 1679.29it/s]
Download complete: : 0.00B [00:00, ?B/s]                                                                                                                                                                                                                                                                                                                      | 0/3 [00:00<?, ?it/s]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 686/686 [00:04<00:00, 142.92it/s, Materializing param=model.vision_tower.vision_model.pre_layrnorm.weight]
=== Breakdown (exact match) ===
color         76/ 143  acc=0.5315
count        223/ 503  acc=0.4433
material     129/ 169  acc=0.7633
shape        109/ 173  acc=0.6301
size         121/ 177  acc=0.6836
yesno        417/ 835  acc=0.4994
ALL        1075/2000  acc=0.5375

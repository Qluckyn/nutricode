data文件下存放了我们对Stable Diffusion模型进行Lora微调的类别和数据

bash_run.sh为Lora微调脚本
datadream.py为基于Datadream的训练代码
util_data.py和utils.py则提供类别处理和随机种子等工具


手部 LoRA 独立流程
=================

手部实验使用 run_hand_lora.sh，不修改也不复用面部 bash_run.sh 的类别和输出目录。
默认读取：

/root/autodl-tmp/data_hand/lora_train_audited/fold_0

四个类别依次为：

0 malnourished_hand_pose01
1 malnourished_hand_pose02
2 normal_hand_pose01
3 normal_hand_pose02

默认只检查配置，不启动训练：

bash run_hand_lora.sh

单类 1 epoch 冒烟测试：

CLASS_IDX=0 EPOCHS=1 RUN_TRAINING=true bash run_hand_lora.sh

按默认 40 epoch 训练四类：

CLASS_IDX=all RUN_TRAINING=true bash run_hand_lora.sh

重要约束：

1. 必须同时提供阶段 A 的 manifest.json，训练前会逐图核对 SHA-256。
2. 手部图片采用保持宽高比、白色补边成正方形、再缩放到 512x512 的预处理。
3. 手部使用固定临床 prompt；面部继续使用原 TEMPLATES_SMALL。
4. 默认仅训练 UNet LoRA，TRAIN_TEXT_ENCODER=false，降低 12 张小样本的过拟合风险。
5. 可通过 CLASS_IDX 指定单类，所有输出按 fold、seed、类别和配置隔离。

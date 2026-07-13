bash_run.sh为生成脚本

generate.py为原始基于DataDream的图像生成代码


手部生成独立流程
================

手部实验使用 run_hand_generate.sh，不修改面部 bash_run.sh，也不写入面部生成目录。

默认仅检查四套阶段 B 权重：

bash run_hand_generate.sh

每类生成 2 张用于阶段 C 验收：

NIPC=2 RUN_GENERATION=true bash run_hand_generate.sh

正式扩大生成数量前，应先人工检查少量结果中的双手数量、手指解剖和姿势。

每张手部生成图都会在对应类别目录的 metadata.jsonl 中记录：类别、姿势、营养状态、
正向 prompt、负向 prompt、独立 seed、LoRA 权重路径、guidance scale 和推理步数。
运行级配置保存在 generation_config.json。已有同名图片默认拒绝覆盖。

当前采用 stage_c_smoke_v2 已验证的详细版 prompt。背景遵循、手部解剖和姿势质量
仍需在阶段 D 继续筛选。



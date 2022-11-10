'''


1.全标签
all_label bert with upan
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/nlpcc_base_bert_all     --task_name allnlpcct5    --do_train     --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir E:/model/nlpcc_base_bert_all/output_dir/

all_label xlnet with upan
python ./examples/run_glue.py     --model_type xlnet     --model_name_or_path F:/model1/nlpcc_base_xlnet_all     --task_name allnlpcct5    --do_train     --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir F:/model1/nlpcc_base_xlnet_all/output_dir/

all_label xlnet with upan test
python ./examples/run_glue.py     --model_type xlnet     --model_name_or_path F:/model1/nlpcc_base_xlnet_all/output_dir/checkpoint-67500     --task_name allnlpcct5    --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir F:/model1/nlpcc_base_xlnet_all/output_dir/checkpoint-67500/test


all_label roberta with upan
python ./examples/run_glue.py     --model_type roberta     --model_name_or_path F:/model1/nlpcc_base_roberta_all     --task_name allnlpcct5    --do_train     --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir F:/model1/nlpcc_base_roberta_all/output_dir/

all_label rpberta test



2.level1标签
bert train
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/nlpcc_base_bert     --task_name nlpcct5    --do_train     --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir E:/model/nlpcc_base_bert/output_dir/
bert test
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/nlpcc_base_bert/epcoh3_21_title    --task_name nlpcct5level1     --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir E:/model/nlpcc_base_bert/epcoh3_21_title
acc = 0.9255142857142857
macro_f1 = 0.7178801980374249
micro_f1 = 0.7692852296527921

bert 4 256 1e-5 10.2 1：24
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/bertbase_lr10_bs4_256_lv1     --task_name nlpcct5level1    --do_train     --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 256     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir E:/model/bertbase_lr10_bs4_256_lv1/output_dir/


roberta test
python ./examples/run_glue.py     --model_type roberta     --model_name_or_path  E:/model/nlpcc_base_roberta/output_dir/checkpoint-33500     --task_name nlpcct5level1     --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir  E:/model/nlpcc_base_roberta/output_dir/checkpoint-33500/

acc = 0.9248761904761905
macro_f1 = 0.7187208026792506
micro_f1 = 0.7659347181008901

xlnet train
python ./examples/run_glue.py     --model_type xlnet     --model_name_or_path  E:/model/nlpcc_base_xlnet/output_dir/checkpoint-54000     --task_name nlpcct5level1    --do_train    --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir  E:/model/nlpcc_base_xlnet/output_dir/54000_more
xlnetbase_lr5_bs2_512_lv1
python ./examples/run_glue.py     --model_type xlnet     --model_name_or_path  E:/model/xlnetbase_lr5_bs2_512_lv1     --task_name nlpcct5level1    --do_train    --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 1024     --per_gpu_eval_batch_size=2       --per_gpu_train_batch_size=2       --learning_rate 5e-6     --num_train_epochs 3.0     --output_dir  E:/model/xlnetbase_lr5_bs2_512_lv1/output_dir/


xlnet test
python ./examples/run_glue.py     --model_type xlnet     --model_name_or_path  E:/model/nlpcc_base_xlnet/output_dir/checkpoint-54000     --task_name nlpcct5level1     --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir  E:/model/nlpcc_base_xlnet/
acc = 0.7144952380952381
macro_f1 = 0.08397597255236608
micro_f1 = 0.11402056980730582

lr=2e-5
acc = 0.9215428571428571
macro_f1 = 0.6715428321689308
micro_f1 = 0.7500303434882873


longformer train 2110
python ./examples/text-classification/run_glue.py       --model_name_or_path  E:/model/nlpcc_base_longformer     --task_name nlpcct5level1   --do_train    --do_eval    --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512      --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4      --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir  E:/model/nlpcc_base_longformer/output_dir/
test
python ./examples/text-classification/run_glue.py       --model_name_or_path  E:/model/nlpcc_base_longformer/output_dir/checkpoint-67500     --task_name nlpcct5level1   --do_eval    --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512      --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4      --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir  E:/model/nlpcc_base_longformer/

acc = 0.9232190476190476
macro_f1 = 0.7028850488262777
micro_f1 = 0.7597592228380714

cl
bert train
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/nlpcc_cl_bert     --task_name nlpcct5level1    --do_train     --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=2       --per_gpu_train_batch_size=2       --learning_rate 5e-6     --num_train_epochs 3.0     --output_dir E:/model/nlpcc_cl_bert/output_dir/

bertclo_lr10_bs4_256_lv1
python ./examples/run_glue.py     --model_type bert     --model_name_or_path F:/model/bertclo_lr10_bs4_256_lv1/output_dir/checkpoint-67500     --task_name nlpcct5level1    --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 256     --per_gpu_eval_batch_size=16       --per_gpu_train_batch_size=2       --learning_rate 5e-6     --num_train_epochs 3.0     --output_dir F:/model/bertclo_lr10_bs4_256_lv1/output_dir/checkpoint-67500/

cl + w
bertclols_lr10_bs4_256_lv1
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/bertclols_lr10_bs4_256_lv1     --task_name nlpcct5level1    --do_train   --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 256     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir E:/model/bertclols_lr10_bs4_256_lv1/output_dir/

python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/bertclols_lr10_bs4_256_lv1/output_dir/checkpoint-67500     --task_name nlpcct5level1    --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 256     --per_gpu_eval_batch_size=2       --per_gpu_train_batch_size=4       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir E:/model/bertclols_lr10_bs4_256_lv1/output_dir/checkpoint-67500/

cl + w1
bertclols1_lr10_bs4_256_lv1
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/bertclols1_lr10_bs4_256_lv1     --task_name nlpcct5level1    --do_train   --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 256     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir E:/model/bertclols1_lr10_bs4_256_lv1/output_dir/


reuters
reuters_bertbase_lr20_bs8_512_lv1
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/bertclols1_lr10_bs4_256_lv1     --task_name reuters    --do_train   --do_eval     --do_lower_case     --data_dir E:/data/reuters     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir E:/model/reuters_bertbase_lr20_bs8_512_lv1/output_dir/
test
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/reuters_bertbase_lr20_bs8_512_lv1/output_dir/checkpoint-2500     --task_name reuters  --do_eval     --do_lower_case     --data_dir E:/data/reuters     --max_seq_length 512     --per_gpu_eval_batch_size=16       --per_gpu_train_batch_size=8       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir E:/model/reuters_bertbase_lr20_bs8_512_lv1/output_dir/checkpoint-2500/

reuters_bertls_lr20_bs8_512_lv1
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/reuters_bertclols_lr20_bs8_256_lv1     --task_name reuters    --do_train   --do_eval     --do_lower_case     --data_dir E:/data/reuters     --max_seq_length 256     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir E:/model/reuters_bertclols_lr20_bs8_256_lv1/output_dir/
test
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/reuters_bertclols_lr20_bs8_256_lv1     --task_name reuters    --do_train    --do_eval    --do_lower_case     --data_dir E:/data/reuters     --max_seq_length 256     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir E:/model/reuters_bertclols_lr20_bs8_256_lv1/output_dir/tt/

res test
robertabase_lr10_bs3_512_lv3
python ./examples/run_glue.py     --model_type roberta     --model_name_or_path F:/model/robertabase_lr10_bs3_512_lvs/output_dir/checkpoint-58500     --task_name allnlpcct5   --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=3       --per_gpu_train_batch_size=3       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir F:/model/robertabase_lr10_bs3_512_lvs/output_dir/checkpoint-58500/

!!训练错误，舍弃
robertabase_lr20_bs8_512_lvs
python ./examples/run_glue.py     --model_type roberta     --model_name_or_path E:/model/robertabase_lr20_bs8_512_lvs/output_dir/checkpoint-33500     --task_name allnlpcct5   --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=3       --per_gpu_train_batch_size=3       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir E:/model/robertabase_lr20_bs8_512_lvs/output_dir/checkpoint-33500/

bertbase_lr20_bs8_512_lv1 ??
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/bertbase_lr20_bs8_512_lv1/epcoh3_21_title     --task_name nlpcct5level1   --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir E:/model/bertbase_lr20_bs8_512_lv1/epcoh3_21_title/

bertbase_lr10_bs3_512_lv1_1 15:43
python ./examples/run_glue.py     --model_type bert     --model_name_or_path F:/model/bertbase_lr10_bs3_512_lv1_1/output_dir/checkpoint-90000     --task_name nlpcct5level1   --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=3       --per_gpu_train_batch_size=3       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir F:/model/bertbase_lr10_bs3_512_lv1_1/output_dir/checkpoint-90000/
v4


xlnetbase_lr20_bs2_512_lv1 15:57
python ./examples/run_glue.py     --model_type xlnet     --model_name_or_path F:/model/xlnetbase_lr20_bs2_512_lv1/output_dir/checkpoint-67500     --task_name nlpcct5level1   --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=2       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir F:/model/xlnetbase_lr20_bs2_512_lv1/output_dir/checkpoint-67500/

bertbase_lr20_bs8_512_lv2
python ./examples/run_glue.py     --model_type bert     --model_name_or_path F:/model/bertbase_lr20_bs8_512_lv2/output_dir     --task_name nlpcct5level2   --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir F:/model/bertbase_lr20_bs8_512_lv2/output_dir/

bertbase_lr10_bs4_512_lv2
python ./examples/run_glue.py     --model_type bert     --model_name_or_path F:/model/bertbase_lr10_bs4_512_lv2/output_dir/checkpoint-67500     --task_name nlpcct5level2   --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=4       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir F:/model/bertbase_lr10_bs4_512_lv2/output_dir/checkpoint-67500/

bertbase_lr10_bs3_512_lv2
python ./examples/run_glue.py     --model_type bert     --model_name_or_path F:/model/bertbase_lr10_bs3_512_lv2/output_dir/checkpoint-90000     --task_name nlpcct5level2   --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=3       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir F:/model/bertbase_lr10_bs3_512_lv2/output_dir/checkpoint-90000/

bertbase_lr20_bs8_512_lvs
python ./examples/run_glue.py     --model_type bert     --model_name_or_path F:/model/bertbase_lr20_bs8_512_lvs/output_dir/checkpoint-33500     --task_name allnlpcct5   --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=4       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir F:/model/bertbase_lr20_bs8_512_lvs/output_dir/checkpoint-33500/

xlnetbase_lr5_bs2_512_lv1
python ./examples/run_glue.py     --model_type xlnet     --model_name_or_path  E:/model/xlnetbase_lr5_bs2_1024_lv1/output_dir/checkpoint-135000     --task_name nlpcct5level1     --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 1024     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=2       --learning_rate 5e-6     --num_train_epochs 3.0     --output_dir  E:/model/xlnetbase_lr5_bs2_1024_lv1/output_dir/checkpoint-135000/



3.level2标签
bert train
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/nlpcc_base_bert_level2    --task_name nlpcct5level2    --do_train     --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir E:/model/nlpcc_base_bert_level2/output_dir/
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/nlpcc_base_bert_level2    --task_name nlpcct5level2    --do_train     --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir E:/model/nlpcc_base_bert_level2/output_dir/
bert test
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/nlpcc_base_bert_level2/output_dir/checkpoint-33500    --task_name nlpcct5level2   --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir E:/model/nlpcc_base_bert_level2/output_dir/
acc = 0.9875153846153846
macro_f1 = 0.12541599284502558
micro_f1 = 0.528444418618165

bert train epoch = 4

python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/nlpcc_base_bert_level2_4    --task_name nlpcct5level2    --do_train     --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir E:/model/nlpcc_base_bert_level2_4/output_dir/
python ./examples/run_glue.py     --model_type bert     --model_name_or_path E:/model/nlpcc_base_bert_level2_4    --task_name nlpcct5level2   --do_eval     --do_lower_case     --data_dir D:/study/nlpcc/traning_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir E:/model/nlpcc_base_bert_level2_4/output_dir/
acc = 0.98699
macro_f1 = 0.1030889123657036
micro_f1 = 0.4905569444862796



vrsion 4
bert_test_for_4
wnli
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/bert_test_for_4     --train_file E:/data/wnli/train.json  --validation_file   E:/data/wnli/validation.json  --max_length 512   --do_train True  --report_to wandb   --per_device_eval_batch_size 4    --learning_rate 1e-5     --num_train_epochs 3   --output_dir E:/model/bert_test_for_4/output_dir/  --with_tracking

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/bert_test_for_4     --train_file E:/data/wnli/train.json  --validation_file   E:/data/wnli/validation.json  --max_length 512  --per_device_eval_batch_size 4    --learning_rate 1e-5     --num_train_epochs 3    --output_dir E:/model/bert_test_for_4/output_dir/

cola
train
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/bert_test_for_4   --do_train True   --train_file E:/data/cola/train.csv  --validation_file   E:/data/cola/test.csv  --max_length 512  --report_to wandb   --per_device_eval_batch_size 4    --learning_rate 1e-5     --num_train_epochs 3   --checkpointing_steps 100 --output_dir E:/model/bert_test_for_4/output_dir/  --with_tracking

reuters test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/reuters_bertclols_lr20_bs8_256_lv1     --train_file E:/data/reuters21578/reuters21578.py --max_length 256  --report_to wandb   --per_device_eval_batch_size 8    --learning_rate 2e-5     --num_train_epochs 3   --checkpointing_steps 100 --output_dir E:/model/bert_test_for_4/output_dir/  --with_tracking

raft
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/bert_test_for_4     --train_file E:/data/raft/raft.py --max_length 256  --report_to wandb   --per_device_eval_batch_size 8    --learning_rate 2e-5     --num_train_epochs 3   --checkpointing_steps 100 --output_dir E:/model/bert_test_for_4/output_dir/  --with_tracking

nlpcc
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/bert_test_for_4     --train_file E:/data/nlpcct5/nlpcct5.py  --max_length 512   --do_train True  --report_to wandb   --per_device_eval_batch_size 4    --learning_rate 1e-5     --num_train_epochs 3   --output_dir E:/model/bert_test_for_4/output_dir/  --with_tracking




E:/data/nlpcct5/nlpcct5.py


test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/bertclols_lr10_bs4_256_lv1/output_dir/checkpoint-67500     --train_file E:/data/nlpcct5/nlpcct5.py  --multicheck False  --report_to wandb   --with_tracking      --max_length 512     --per_device_eval_batch_size 4    --num_train_epochs 3     --output_dir E:/model/bertclols_lr10_bs4_256_lv1/output_dir/checkpoint-67500/
train
bertbase_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/bertbase_lr20_bs8_256_lv1     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True  --report_to wandb   --with_tracking      --max_length 256    --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir E:/model/bertbase_lr20_bs8_256_lv1/output_dir/  --checkpointing_steps 100
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/bertbase_lr20_bs8_256_lv1/output_dir     --train_file E:/data/nlpcct5/nlpcct5.py     --max_length 256    --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir E:/model/bertbase_lr20_bs8_256_lv1/output_dir/


xlnetbase_lr5_bs2_1024_lv1_1
E:/model/xlnetbase_lr5_bs2_1024_lv1_1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/xlnetbase_lr5_bs2_1024_lv1_1     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True  --report_to wandb   --with_tracking     --learning_rate 5e-6    --max_length 1024 --per_device_train_batch_size 2   --per_device_eval_batch_size 2    --num_train_epochs 3     --output_dir E:/model/xlnetbase_lr5_bs2_1024_lv1_1  --checkpointing_steps 1000

bertfl_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True  --report_to wandb   --with_tracking     --learning_rate 2e-5    --max_length 512 --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir E:/model/transformers4/bertfl_lr20_bs8_256_lv1  --checkpointing_steps 1000

bertflbce_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True  --report_to wandb   --with_tracking     --learning_rate 2e-5    --max_length 512 --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir E:/model/transformers4/bertflbce_lr20_bs8_256_lv1  --checkpointing_steps 1000


python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/bert_for_test     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True   --max_length 64    --per_device_eval_batch_size 16    --num_train_epochs 3     --output_dir E:/model/transformers4/bert_for_test/output_dir/  --checkpointing_steps 100

bertrd50_lr10_bs4_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True  --report_to wandb   --with_tracking     --learning_rate 1e-5    --max_length 256 --per_device_train_batch_size 4   --per_device_eval_batch_size 4    --num_train_epochs 3     --output_dir E:/model/transformers4/bertrd50_lr10_bs4_256_lv1/  --checkpointing_steps 1000

bertrd30_lr10_bs4_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True  --report_to wandb   --with_tracking     --learning_rate 1e-5    --max_length 256 --per_device_train_batch_size 4   --per_device_eval_batch_size 4    --num_train_epochs 3     --output_dir E:/model/transformers4/bertrd30_lr10_bs4_256_lv1/  --checkpointing_steps 1000

bertrdrop1_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True  --report_to wandb   --with_tracking     --learning_rate 2e-5    --max_length 256 --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir E:/model/transformers4/bertrdrop1_lr20_bs8_256_lv1/  --checkpointing_steps 1000

bertrdropdblossntrfl_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True  --report_to wandb   --with_tracking     --learning_rate 2e-5    --max_length 256 --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir E:/model/transformers4/berrdropdblossntrfl_lr20_bs8_256_lv1/  --checkpointing_steps 1000


bertrdropsuperloss_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5.py --child_tune --do_train True    --learning_rate 2e-5    --max_length 256 --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir E:/model/transformers4/bertrdropsuperloss_lr20_bs8_256_lv1/


python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5.py --child_tune --do_train True    --learning_rate 2e-5    --max_length 256 --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir E:/model/transformers4/bertrdropsuperloss_lr20_bs8_256_lv1_1/



bertcdloss_md_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True  --report_to wandb   --with_tracking     --learning_rate 2e-5    --max_length 256 --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir E:/model/transformers4/bertcdloss_md_lr20_bs8_256_lv1/  --checkpointing_steps 1000
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/bertcdloss_md_lr20_bs8_256_lv1     --train_file E:/data/nlpcct5/nlpcct5.py     --learning_rate 2e-5    --max_length 256 --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir E:/model/transformers4/bertcdloss_md_lr20_bs8_256_lv1/


bertbase_lr20_bs8_256_lv2
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_lv2.py  --do_train True  --learning_rate 2e-5  --report_to wandb   --with_tracking      --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir E:/model/transformers4/bertbase_lr20_bs8_256_lv2/  --checkpointing_steps 1000




train for test
nlpcc
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/bert_for_test     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True   --max_length 64  --per_device_train_batch_size 16   --per_device_eval_batch_size 16     --num_train_epochs 3     --output_dir E:/model/transformers4/bert_for_test/output_dir/  --checkpointing_steps 100

python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/bert_for_test     --train_file E:/data/nlpcct5/nlpcct5.py   --max_length 64  --per_device_train_batch_size 16   --per_device_eval_batch_size 16     --num_train_epochs 3     --output_dir E:/model/transformers4/bert_for_test/output_dir/  --checkpointing_steps 100

train for simcse
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5.py   --max_length 256  --per_device_train_batch_size 8   --per_device_eval_batch_size 8   --train_mode simcse  --do_train True --report_to wandb --with_tracking  --num_train_epochs 3     --output_dir E:/model/transformers4/nlpcc_simcse_bert/  --checkpointing_steps 1000

train for simcse1 test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5.py   --max_length 64  --per_device_train_batch_size 32   --per_device_eval_batch_size 32   --train_mode simcse  --do_train True --num_train_epochs 3     --output_dir E:/model/transformers4/nlpcc_simcse_bert1/  --checkpointing_steps 100

train for simcse2
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5.py   --max_length 256  --per_device_train_batch_size 8   --per_device_eval_batch_size 8   --train_mode simcse  --do_train True --report_to wandb --with_tracking  --num_train_epochs 3     --output_dir E:/model/transformers4/nlpcc_simcse_bert2/  --checkpointing_steps 1000

train for simcse3
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5.py   --max_length 256  --per_device_train_batch_size 8   --per_device_eval_batch_size 8   --train_mode simcse  --do_train True --report_to wandb --with_tracking  --num_train_epochs 3     --output_dir E:/model/transformers4/nlpcc_simcse_bert3/  --checkpointing_steps 1000

train for simcse_sup
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_rand.py   --max_length 256  --per_device_train_batch_size 4   --per_device_eval_batch_size 4   --train_mode simcse_sup  --do_train True --report_to wandb --with_tracking  --num_train_epochs 3     --output_dir E:/model/transformers4/nlpcc_simcse_sup_bert/  --checkpointing_steps 1000

train for simcse_sup1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_rand.py   --max_length 256  --per_device_train_batch_size 8   --per_device_eval_batch_size 8   --train_mode simcse_sup  --do_train True   --num_train_epochs 3     --output_dir E:/model/transformers4/nlpcc_simcse_sup_bert_1/  --checkpointing_steps 1000

train for simcse_sup2 sim together
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_rand_sim_together.py   --max_length 256  --per_device_train_batch_size 8   --per_device_eval_batch_size 8   --train_mode simcse_sup  --do_train True   --num_train_epochs 3     --output_dir E:/model/transformers4/nlpcc_simcse_sup_bert_2/  --checkpointing_steps 1000

train for simcse_sup3 sim together
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_rand_sim_together.py   --max_length 256  --per_device_train_batch_size 8   --per_device_eval_batch_size 8   --train_mode simcse_sup  --do_train True   --num_train_epochs 3     --output_dir E:/model/transformers4/nlpcc_simcse_sup_bert_3/  --checkpointing_steps 1000


bertsimcse_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/nlpcc_simcse_bert     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True    --learning_rate 2e-5    --max_length 256 --per_device_train_batch_size 8   --per_device_eval_batch_size 8  --report_to wandb --with_tracking   --num_train_epochs 3     --output_dir E:/model/transformers4/bertsimcse_lr20_bs8_256_lv1/  --checkpointing_steps 1000

bertsimcse_lr20_bs8_256_lv1_1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/nlpcc_simcse_bert2     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True    --learning_rate 2e-5    --max_length 256 --per_device_train_batch_size 8   --per_device_eval_batch_size 8  --report_to wandb --with_tracking   --num_train_epochs 3     --output_dir E:/model/transformers4/bertsimcse_lr20_bs8_256_lv1_2/  --checkpointing_steps 1000


bertsimcsesimt_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/nlpcc_simcse_sup_bert_2     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True    --learning_rate 2e-5    --max_length 256 --per_device_train_batch_size 8   --per_device_eval_batch_size 8  --report_to wandb --with_tracking   --num_train_epochs 3     --output_dir E:/model/transformers4/bertsimcsesimt_lr20_bs8_256_lv1/  --checkpointing_steps 1000

bertsimcsesimt_lr20_bs8_256_lv1_1 1000epochtrain
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/nlpcc_simcse_sup_bert_2/step_1000     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True    --learning_rate 2e-5    --max_length 256 --per_device_train_batch_size 8   --per_device_eval_batch_size 8  --report_to wandb --with_tracking   --num_train_epochs 3     --output_dir E:/model/transformers4/bertsimcsesimt_lr20_bs8_256_lv1_1/  --checkpointing_steps 1000


bertsimcsesimt_lr20_bs8_256_lv1_2 10000epochtrain
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/nlpcc_simcse_sup_bert_2/step_10000     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True    --learning_rate 2e-5    --max_length 256 --per_device_train_batch_size 8   --per_device_eval_batch_size 8  --report_to wandb --with_tracking   --num_train_epochs 3     --output_dir E:/model/transformers4/bertsimcsesimt_lr20_bs8_256_lv1_2/  --checkpointing_steps 1000


bertsimcsesimt_lr20_bs8_256_lv1_3 0.15 8000epochtrain
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/nlpcc_simcse_sup_bert_3/step_8000     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True    --learning_rate 2e-5    --max_length 256 --per_device_train_batch_size 8   --per_device_eval_batch_size 8  --report_to wandb --with_tracking   --num_train_epochs 3     --output_dir E:/model/transformers4/bertsimcsesimt_lr20_bs8_256_lv1_3/  --checkpointing_steps 1000




reuters

python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/bert_for_test     --train_file E:/data/reuters/reuters.py  --do_train True   --max_length 512  --per_device_train_batch_size 8   --per_device_eval_batch_size 8      --num_train_epochs 3     --output_dir E:/model/transformers4/bert_for_test/output_dir/  --checkpointing_steps 1000
E:\model\transformers4\bert_for_test\output_dir
test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/bert_for_test/output_dir     --train_file E:/data/reuters/reuters.py   --max_length 256  --per_device_train_batch_size 8   --per_device_eval_batch_size 8      --num_train_epochs 3     --output_dir E:/model/transformers4/bert_for_test/output_dir/tt



find for threshold
bertbase_lr20_bs8_256_lv1 0.38
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/bertbase_lr20_bs8_256_lv1/output_dir     --train_file E:/data/nlpcct5/nlpcct5.py   --max_length 256  --per_device_train_batch_size 8   --per_device_eval_batch_size 8      --num_train_epochs 3     --output_dir E:/model/transformers4/bertbase_lr20_bs8_256_lv1/output_dir

bertrdrop1_lr20_bs8_256_lv1_1 0.38
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/bertrdrop1_lr20_bs8_256_lv1_1     --train_file E:/data/nlpcct5/nlpcct5.py   --max_length 256  --per_device_train_batch_size 8   --per_device_eval_batch_size 8      --num_train_epochs 3     --output_dir E:/model/transformers4/bertrdrop1_lr20_bs8_256_lv1_1

bertrdrop1_lr20_bs8_256_lv1 0.37
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/bertrdrop1_lr20_bs8_256_lv1     --train_file E:/data/nlpcct5/nlpcct5.py   --max_length 256  --per_device_train_batch_size 8   --per_device_eval_batch_size 8      --num_train_epochs 3     --output_dir E:/model/transformers4/bertrdrop1_lr20_bs8_256_lv1

bertflbce_lr20_bs8_256_lv1 0.46
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/bertflbce_lr20_bs8_256_lv1     --train_file E:/data/nlpcct5/nlpcct5.py   --max_length 256  --per_device_train_batch_size 8   --per_device_eval_batch_size 8      --num_train_epochs 3     --output_dir E:/model/transformers4/bertflbce_lr20_bs8_256_lv1

bertfl_lr20_bs8_256_lv1 0.45
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/transformers4/bertfl_lr20_bs8_256_lv1     --train_file E:/data/nlpcct5/nlpcct5.py   --max_length 256  --per_device_train_batch_size 8   --per_device_eval_batch_size 8      --num_train_epochs 3     --output_dir E:/model/transformers4/bertfl_lr20_bs8_256_lv1



zhihu_bertfl_lr40_bs16_128
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 4e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/zhihu_bertfl_lr40_bs16_128   --checkpointing_steps epoch
--report_to wandb --with_tracking

python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 4e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/zhihu_bertfl_lr40_bs16_128   --checkpointing_steps epoch

train for simcse
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/white_model/bert     --train_file E:/data/zhihu/zhihu.py   --max_length 128  --per_device_train_batch_size 16   --per_device_eval_batch_size 16   --train_mode simcse  --do_train True  --num_train_epochs 3     --output_dir E:/model/transformers4/zhihu_simcse_bert/  --checkpointing_steps 1503



E:/model/transformers4/nlpcc_simcse_sup_bert_1
no wei
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/nlpcc_simcse_sup_bert_1     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True   --report_to wandb   --with_tracking   --max_length 256    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3    --output_dir E:/model/transformers4/simcse_sup_bertbase_lr20_bs8_256/    --checkpointing_steps 1000
{'suset_accuracy': 0.1524, 'accuracy': 0.6318224786324755, 'precision': 0.7519471428571429, 'recall': 0.7908550183150183, 'f1': 0.7520404484404484, 'micro-precision': 0.7316728301065912, 'micro-recall': 0.
7721090387374462, 'micro-f1': 0.75134727612878, 'threshold': 0.4}

wei
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/nlpcc_simcse_sup_bert_1     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True   --report_to wandb   --with_tracking   --max_length 256    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3    --output_dir E:/model/transformers4/simcse_sup_bertbasew_lr20_bs8_256/    --checkpointing_steps 1000



E:/model/transformers4/zhihu_simcse_bert
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/zhihu_simcse_bert     --train_file E:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 4e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/zhihu_simcse_bertfl_lr40_bs16_128   --checkpointing_steps epoch


hm12
berthm12_lr20_bs8_256_lv12
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12   --checkpointing_steps 2000

berthm12_lr20_bs8_256_lv12_1_1 x
{'suset_accuracy': 0.0, 'accuracy': 0.14325285601420046, 'precision': 0.352078253968254, '
recall': 0.17995011496440066, 'f1': 0.23136269808762153, 'micro-precision': 0.3628894745000775, 'micro-recall': 0.1761872506961692, 'micro-f1': 0.
23720741716485969}
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_1_1   --checkpointing_steps 2000

berthm12_lr20_bs8_256_lv12_1_2
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_1_2   --checkpointing_steps 2000


berthm12_lr20_bs8_256_lv12_1_4
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_1_4   --checkpointing_steps 2000


berthm12_lr20_bs8_256_lv12_1_5
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_1_4   --checkpointing_steps 2000

berthm12_lr20_bs8_256_lv12_1_6
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_1_6   --checkpointing_steps 2000


berthm12_lr20_bs8_256_lv12_1_7
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_1_7   --checkpointing_steps 2000

berthm12_lr20_bs8_256_lv12_1_8
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_1_8   --checkpointing_steps 2000


berthm12_lr20_bs8_256_lv12_1_9
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_1_9   --checkpointing_steps 2000



berthm12_lr20_bs8_256_lv12_1_10
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/nlpcc_simcse_bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_1_10   --checkpointing_steps 2000


berthm12_lr20_bs8_256_lv12_1_14
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_1_14   --checkpointing_steps 2000

berthm12_lr20_bs8_256_lv12_1_15
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_1_15   --checkpointing_steps 2000

berthm12_lr20_bs8_256_lv12_1_16
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_1_16   --checkpointing_steps 2000


berthm12_lr20_bs8_256_lv12_1_17
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True    --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_1_17   --checkpointing_steps 2000

berthm12_lr20_bs8_512_lv12_1_18
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_512_lv12_1_18   --checkpointing_steps 2000

berthm12_lr20_bs8_512_lv12_1_20
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_512_lv12_1_20   --checkpointing_steps 2000


berthm123_lr20_bs8_512_lv123
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm123.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm123_lr20_bs8_512_lv123   --checkpointing_steps 2000

sciberthm123_lr20_bs8_256_lv123_1 1.0版本
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/nlpcct5/nlpcct5_hm123.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_lr20_bs8_512_lv123   --checkpointing_steps 2000


scibert123_lr20_bs8_256_lv123_1_3
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/nlpcct5/nlpcct5_hm123.py   --max_length 256   --do_train True  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_3   --checkpointing_steps 2000


scibert123_lr20_bs8_256_lv123_1_7
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/nlpcct5/nlpcct5_hm123.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_7   --checkpointing_steps 2000



sciberthm12_lr20_bs8_256_wos_test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm12_lr20_bs8_256_wos_test   --checkpointing_steps 2

sciberthm12_lr20_bs8_256_wos_simcse_sup_test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/wos/wos_rand_sim_together.py   --max_length 256   --do_train True --train_mode simcse_sup   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm12_lr20_bs8_256_wos_simcse_sup_test   --checkpointing_steps 2


scibert123_lr20_bs8_256_lv123_1_8_simcse_base
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/wos/wos_rand_sim_together.py   --max_length 256   --do_train True --train_mode simcse_sup   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_8_simcse_base   --checkpointing_steps 1000

scibert123_lr20_bs8_256_lv123_1_10_simcse_base
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/wos/wos_rand_sim_together.py   --max_length 256   --do_train True --train_mode simcse_sup   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 1   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_10_simcse_base   --checkpointing_steps 1000



scibert123_lr20_bs8_256_lv123_1_8
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_8_simcse_base/step_4000     --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_8   --checkpointing_steps 1000
test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_8     --train_file E:/data/wos/wos.py   --max_length 256   --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_8   --checkpointing_steps 1000


scibert123_lr20_bs8_256_lv123_1_9_simcse_base
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/wos/wos_rand_sim_together.py   --max_length 256   --do_train True --train_mode simcse_sup   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_9_simcse_base   --checkpointing_steps 1000

scibert123_lr20_bs8_256_lv123_1_9
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_9_simcse_base/     --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_9   --checkpointing_steps 1000 --ignore_mismatched_sizes


scibert123_lr20_bs8_256_lv123_1_10
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_8_simcse_base/step_4000     --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_10   --checkpointing_steps 1000  --ignore_mismatched_sizes




hm12 debug
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_test   --checkpointing_steps 2

python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm123.py   --max_length 256   --do_train True  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm123_lr20_bs8_512_lv123   --checkpointing_steps 2


'''

'''
电脑2

all_label xlnet
python ./examples/run_glue.py     --model_type xlnet     --model_name_or_path D:/model/nlpcc_base_xlnet_all     --task_name allnlpcct5    --do_train     --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=2       --per_gpu_train_batch_size=2       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir D:/model/nlpcc_base_xlnet_all/output_dir/

all_label xlnet test 
python ./examples/run_glue.py     --model_type xlnet     --model_name_or_path D:/model/nlpcc_base_xlnet_all     --task_name allnlpcct5    --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=2       --per_gpu_train_batch_size=2       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir D:/model/nlpcc_base_xlnet_all/

notrain xlnet test
python ./examples/run_glue.py     --model_type xlnet     --model_name_or_path D:/model/nlpcc_base_xlnet_all     --task_name allnlpcct5    --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=2       --per_gpu_train_batch_size=2       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir D:/model/nlpcc_base_xlnet_all/



python ./examples/run_glue.py     --model_type roberta     --model_name_or_path D:/model/nlpcc_roberta_base_all     --task_name allnlpcct5    --do_train     --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=2       --per_gpu_train_batch_size=2       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir D:/model/nlpcc_roberta_base_all/output_dir/



lavel1_label xlnet
python ./examples/run_glue.py     --model_type xlnet     --model_name_or_path D:/model/nlpcc_base_xlnet     --task_name nlpcct5level1     --do_train   --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=2       --per_gpu_train_batch_size=2       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir D:/model/nlpcc_base_xlnet/output_dir1/

lavel1_label xlnet test
python ./examples/run_glue.py     --model_type xlnet     --model_name_or_path D:/model/nlpcc_base_xlnet/output_dir/checkpoint-54000     --task_name nlpcct5level1     --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=2       --per_gpu_train_batch_size=2       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir D:/model/nlpcc_base_xlnet/output_dir/checkpoint-54000/


lavel1_label bert 
python ./examples/run_glue.py     --model_type bert     --model_name_or_path D:/model/nlpcc_bert_base/output_dir/checkpoint-33000     --task_name nlpcct5level1    --do_train     --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=3       --per_gpu_train_batch_size=3       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir D:/model/nlpcc_bert_base/output_dir/33000-more/

level1_label bert test
python ./examples/run_glue.py     --model_type bert     --model_name_or_path D:/model/nlpcc_bert_base_level1_3/output_dir/checkpoint-90000     --task_name nlpcct5level1     --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=3       --per_gpu_train_batch_size=3       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir D:/model/nlpcc_bert_base_level1_3/output_dir/checkpoint-90000/


level1_label_256 256size train
python ./examples/run_glue.py     --model_type bert     --model_name_or_path D:/model/bertbase_lr10_bs3_256_lv1     --task_name nlpcct5level1     --do_train    --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 256     --per_gpu_eval_batch_size=3       --per_gpu_train_batch_size=3       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir D:/model/bertbase_lr10_bs3_256_lv1/output_dir/


bertcl_lr5_bs2_256_lv1
python ./examples/run_glue.py     --model_type bert     --model_name_or_path D:/model/bertcl_lr5_bs2_256_lv1     --task_name nlpcct5level1    --do_train     --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 256     --per_gpu_eval_batch_size=2       --per_gpu_train_batch_size=2       --learning_rate 5e-6     --num_train_epochs 3.0     --output_dir D:/model/bertcl_lr5_bs2_256_lv1/output_dir/
continue train 
python ./examples/run_glue.py     --model_type bert     --model_name_or_path D:/model/bertcl_lr5_bs4_256_lv1/output_dir/checkpoint-123500     --task_name nlpcct5level1     --do_train    --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 256     --per_gpu_eval_batch_size=2       --per_gpu_train_batch_size=2       --learning_rate 5e-6     --num_train_epochs 3.0     --output_dir D:/model/bertcl_lr5_bs2_256_lv1/output_dir/123500more/

test
python ./examples/run_glue.py     --model_type bert     --model_name_or_path D:/model/bertcl_lr5_bs2_256_lv1/output_dir/checkpoint-135000     --task_name nlpcct5level1    --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 256     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=2       --learning_rate 5e-6     --num_train_epochs 3.0     --output_dir D:/model/bertcl_lr5_bs2_256_lv1/output_dir/checkpoint-135000/


bertclo_lr10_bs4_256_lv1
python ./examples/run_glue.py     --model_type bert     --model_name_or_path D:/model/bertclo_lr10_bs4_256_lv1     --task_name nlpcct5level1    --do_train     --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 256     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir D:/model/bertclo_lr10_bs4_256_lv1/output_dir/

bertclo_lr20_bs8_256_lv1
python ./examples/run_glue.py     --model_type bert     --model_name_or_path D:/model/bertclo_lr10_bs4_256_lv1     --task_name nlpcct5level1    --do_train     --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 256     --per_gpu_eval_batch_size=8       --per_gpu_train_batch_size=8       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir D:/model/bertclo_lr20_bs8_256_lv1/output_dir/

bertclo_lr10_bs4_256_lv1
python ./examples/run_glue.py     --model_type bert     --model_name_or_path D:/model/bertclo_lr10_bs4_512_lv1     --task_name nlpcct5level1    --do_train     --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 256     --per_gpu_eval_batch_size=4       --per_gpu_train_batch_size=4       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir D:/model/bertclo_lr10_bs4_256_lv1/output_dir/


level2_label bert bese
python ./examples/run_glue.py     --model_type bert     --model_name_or_path D:/model/nlpcc_bert_base_level2_3    --task_name nlpcct5level2    --do_train     --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=3       --per_gpu_train_batch_size=3       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir D:/model/nlpcc_bert_base_level2_3/output_dir/
python ./examples/run_glue.py     --model_type bert     --model_name_or_path D:/model/nlpcc_bert_base_level2_3/output_dir/checkpoint-56500    --task_name nlpcct5level2    --do_train     --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=3       --per_gpu_train_batch_size=3       --learning_rate 2e-5     --num_train_epochs 3.0     --output_dir D:/model/nlpcc_bert_base_level2_3/output_dir/more_56500/


level3标签
bert train
python ./examples/run_glue.py     --model_type bert     --model_name_or_path D:/model/nlpcc_bert_base_level3_3    --task_name nlpcct5level3    --do_train     --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=3       --per_gpu_train_batch_size=3       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir D:/model/nlpcc_bert_base_level3_3/output_dir/

bert 



bertbase_lr10_bs3_512_lv3 
python ./examples/run_glue.py     --model_type bert     --model_name_or_path F:/model/bertbase_lr10_bs3_512_lv3/output_dir/checkpoint-90000    --task_name nlpcct5level3    --do_eval     --do_lower_case     --data_dir D:/data/nlpcct5/training_datasets     --max_seq_length 512     --per_gpu_eval_batch_size=3       --per_gpu_train_batch_size=3       --learning_rate 1e-5     --num_train_epochs 3.0     --output_dir F:/model/bertbase_lr10_bs3_512_lv3/output_dir/checkpoint-90000/

bertrdrop50_lr10_bs4_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_uncased     --train_file D:/data/nlpcct5/nlpcct5.py  --max_length 256   --do_train True  --report_to wandb    --with_tracking --per_device_eval_batch_size 4  --per_device_train_batch_size 4  --learning_rate 1e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/bertrdrop50_lr20_bs8_256_lv1/ --checkpointing_steps 1000

bertcb_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_uncased     --train_file D:/data/nlpcct5/nlpcct5.py  --max_length 256   --do_train True  --report_to wandb    --with_tracking --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/bertcb_lr20_bs8_256_lv1/ --checkpointing_steps 1000

bertntrfl_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_uncased     --train_file D:/data/nlpcct5/nlpcct5.py  --max_length 256   --do_train True  --report_to wandb    --with_tracking --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/bertntrfl_lr20_bs8_256_lv1/ --checkpointing_steps 1000


bertcbloss_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_uncased     --train_file D:/data/nlpcct5/nlpcct5.py  --max_length 256   --do_train True  --report_to wandb    --with_tracking --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/bertcbloss_lr20_bs8_256_lv1/ --checkpointing_steps 1000

bertcblossrrop_l10_bs4_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_uncased     --train_file D:/data/nlpcct5/nlpcct5.py  --max_length 256   --do_train True  --report_to wandb    --with_tracking --per_device_eval_batch_size 4  --per_device_train_batch_size 4  --learning_rate 1e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/bertcblossrrop_l10_bs4_256_lv1/ --checkpointing_steps 1000

bertfl_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_uncased     --train_file D:/data/nlpcct5/nlpcct5.py  --max_length 256   --do_train True  --report_to wandb    --with_tracking --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/bertfl_lr20_bs8_256_lv1/ --checkpointing_steps 1000


bertrdropcblossntrdbloss_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_uncased     --train_file D:/data/nlpcct5/nlpcct5.py  --max_length 256   --do_train True  --report_to wandb    --with_tracking --per_device_eval_batch_size 4  --per_device_train_batch_size 4  --learning_rate 1e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/bertrdropcblossntrdbloss_lr20_bs8_256_lv1/ --checkpointing_steps 1000


zhihu_bertbase_lr40_bs16_128 
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_chinese     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 4e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertbase_lr40_bs16_128   --checkpointing_steps epoch


zhihu_bertrdrop_lr20_bs8_128 3500
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_chinese     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertrdrop_lr20_bs8_128   --checkpointing_steps epoch




zhihu_bertwrdrop_lr20_bs8_128 0.5kl w
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_chinese     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertwrdrop_lr20_bs8_128   --checkpointing_steps epoch
{'accuracy': 0.7887323943661971, 'precision': 0.47058823529411764, 'recall': 0.5714285714285714, 'f1': 0.5161290322580646}

zhihu_bertwrdrop_lr20_bs8_128_1 1kl w
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_chinese     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertwrdrop_lr20_bs8_128_1   --checkpointing_steps epoch
{'accuracy': 0.7887323943661971, 'precision': 0.4722222222222222, 'recall': 0.6071428571428571, 'f1': 0.53125}

zhihu_bertwrdrop_lr20_bs8_128_2 0.1kl w
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_chinese     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertwrdrop_lr20_bs8_128_2   --checkpointing_steps epoch
{'accuracy': 0.7992957746478874, 'precision': 0.49206349206349204, 'recall': 0.5535714285714286, 'f1': 0.5210084033613446}

zhihu_bertbase_simcse_lr80_bs16_128_3 a=0.5 rdrop w
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_chinese     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertwrdrop_lr20_bs8_128_3   --checkpointing_steps epoch
{'accuracy': 0.7922535211267606, 'precision': 0.4782608695652174, 'recall': 0.5892857142857143, 'f1': 0.5279999999999999}

zhihu_bertwrdrop_lr20_bs8_128_4 1652 rdrop w
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_chinese     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertwrdrop_lr20_bs8_128_4   --checkpointing_steps epoch
{'accuracy': 0.8063380281690141, 'precision': 0.5081967213114754, 'recall': 0.5535714285714286, 'f1': 0.5299145299145298}

zhihu_bertrdrop_lr20_bs8_128_5  a=100 rdrop w
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_chinese     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertwrdrop_lr20_bs8_128_5   --checkpointing_steps epoch

zhihu_simcse
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_chinese     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True  --train_mode simcse  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_simcse_bert   --checkpointing_steps 753

zhihu_simcse_1 1ep
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_chinese     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True  --train_mode simcse  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 1e-5     --num_train_epochs 1   --output_dir D:/model/transformers4/zhihu_simcse_bert_1   --checkpointing_steps 501

zhihu_simcse_2 1ep 0.001
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_chinese     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True  --train_mode simcse  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 1e-5     --num_train_epochs 1   --output_dir D:/model/transformers4/zhihu_simcse_bert_2   --checkpointing_steps 501

zhihu_simcse_3 1ep 0.05 flavg
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_chinese     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True  --train_mode simcse  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 1e-5     --num_train_epochs 1   --output_dir D:/model/transformers4/zhihu_simcse_bert_3   --checkpointing_steps 501


# simcse_sup
zhihu_simcse_sup_1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/chinesebert     --train_file E:\data\zhihu\zhihu_rand_sim_together.py   --max_length 128   --do_train True  --train_mode simcse_sup  --per_device_eval_batch_size 32  --per_device_train_batch_size 32  --learning_rate 8e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/zhihu_simcse_sup_1   --checkpointing_steps 378

zhihu_simcse_sup_2
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/chinesebert     --train_file E:\data\zhihu\zhihu_rand_sim_together.py   --max_length 128   --do_train True  --train_mode simcse_sup  --per_device_eval_batch_size 32  --per_device_train_batch_size 32  --learning_rate 4e-5     --num_train_epochs 1   --output_dir E:/model/transformers4/zhihu_simcse_sup_2   --checkpointing_steps 378


zhihu_bertbase_simcse_sup_lr80_bs32_128
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/zhihu_simcse_sup_1     --train_file E:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 32  --per_device_train_batch_size 32  --learning_rate 8e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/zhihu_bertbase_simcse_sup_lr80_bs32_128   --checkpointing_steps epoch



zhihu_bertbase_simcse_sup_lr80_bs32_128_1 学习率调整
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/zhihu_simcse_sup_1     --train_file E:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 32  --per_device_train_batch_size 32  --learning_rate 4e-5     --num_train_epochs 1   --output_dir D:/model/transformers4/zhihu_bertbase_simcse_sup_lr80_bs32_128_1   --checkpointing_steps epoch
{'accuracy': 0.8133802816901409, 'precision': 0.5238095238095238, 'recall': 0.5892857142857143, 'f1': 0.5546218487394958}

zhihu_bertbase_simcse_sup_lr80_bs32_128_2 学习率调整 epoch调整
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/zhihu_simcse_sup_1     --train_file E:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 32  --per_device_train_batch_size 32  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertbase_simcse_sup_lr80_bs32_128_2   --checkpointing_steps epoch
{'accuracy': 0.7992957746478874, 'precision': 0.4878048780487805, 'recall': 0.35714285714285715, 'f1': 0.41237113402061853} 


zhihu_bertbase_simcse_sup_lr40_bs32_128_3 simcse_bert2
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/zhihu_simcse_sup_2     --train_file E:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 32  --per_device_train_batch_size 32  --learning_rate 4e-5     --num_train_epochs 1   --output_dir E:/model/transformers4/zhihu_bertbase_simcse_sup_lr40_bs32_128_3   --checkpointing_steps epoch


zhihu_bertbase_simcse_sup_lr40_bs32_128_4 
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/chinesebert     --train_file E:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 32  --per_device_train_batch_size 32  --learning_rate 4e-5     --num_train_epochs 1   --output_dir E:/model/transformers4/zhihu_bertbase_simcse_sup_lr40_bs32_128_4   --checkpointing_steps epoch


zhihu_bertbase_simcse_lr40_bs16_128
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/transformers4/zhihu_simcse_bert     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 4e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertbase_simcse_lr40_bs16_128   --checkpointing_steps epoch

zhihu_bertbase_simcse_lr80_bs16_128
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/transformers4/zhihu_simcse_bert     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 8e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertbase_simcse_lr80_bs16_128   --checkpointing_steps epoch





zhihu_bertbase_simcse_lr40_bs16_128_1 1ep
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/transformers4/zhihu_simcse_bert_1     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 4e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertbase_simcse_lr40_bs16_128_1   --checkpointing_steps epoch

zhihu_bertbase_simcse_lr40_bs16_128_2 1ep 0.001 
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/transformers4/zhihu_simcse_bert_2     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 4e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertbase_simcse_lr40_bs16_128_2   --checkpointing_steps epoch
{'accuracy': 0.7816901408450704, 'precision': 0.4642857142857143, 'recall': 0.6964285714285714, 'f1': 0.5571428571428572}



zhihu_bertrdrop_simcse_lr20_bs8_128_2 1ep 0.001 nowei
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/transformers4/zhihu_simcse_bert_2     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertrdrop_simcse_lr20_bs8_128   --checkpointing_steps epoch
{'accuracy': 0.8133802816901409, 'precision': 0.5245901639344263, 'recall': 0.5714285714285714, 'f1': 0.5470085470085471}

D:/model/transformers4/zhihu_simcse_bert_3 1ep 0.5 nowei  flavg
zhihu_bertbase_simcse_lr40_bs16_128_3
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/transformers4/zhihu_simcse_bert_3     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 4e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertbase_simcse_lr40_bs16_128_3   --checkpointing_steps epoch


zhihu_bertbase_simcse_lr40_bs16_128_4 weight
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/transformers4/zhihu_simcse_bert_3     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 4e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertbase_simcse_lr40_bs16_128_4   --checkpointing_steps epoch
{'accuracy': 0.7359154929577465, 'precision': 0.39080459770114945, 'recall': 0.6071428571428571, 'f1': 0.47552447552447547}


zhihu_bertbase_simcse_lr40_bs16_128_4 weight epoch 6
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/transformers4/zhihu_simcse_bert_3     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 4e-5     --num_train_epochs 6   --output_dir D:/model/transformers4/zhihu_bertbase_simcse_lr40_bs16_128_5   --checkpointing_steps epoch
{'accuracy': 0.7922535211267606, 'precision': 0.4727272727272727, 'recall': 0.4642857142857143, 'f1': 0.4684684684684684}

zhihu_bertbase_simcse_lr40_bs16_128_5 weight 0.001 ep1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/transformers4/zhihu_simcse_bert_2     --train_file D:/data/zhihu/zhihu.py   --max_length 128   --do_train True    --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 4e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/zhihu_bertbase_simcse_lr40_bs16_128_5   --checkpointing_steps epoch
{'accuracy': 0.7640845070422535, 'precision': 0.43956043956043955, 'recall': 0.7142857142857143, 'f1': 0.54421768707483}



'''
'''
server
xlnet_lr10_bs4_1024_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path /opt/data/yanyu/xlnet_lr10_bs4_1024_lv1     --train_file /opt/data/yanyu//nlpcct5/nlpcct5.py  --do_train True  --report_to wandb   --with_tracking     --learning_rate 1e-5    --max_length 1024 --per_gpu_train_batch_size 4   --per_device_eval_batch_size 4    --num_train_epochs 3     --output_dir /opt/data/yanyu/xlnet_lr10_bs4_1024_lv1/output_dir/  --checkpointing_steps 1000

bertrd
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path /opt/data/yanyu/white_model/bert_base_uncased     --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5.py   --max_length 256   --do_train True  --report_to tensorboard    --with_tracking --per_device_eval_batch_size 4  --per_device_train_batch_size 4  --learning_rate 1e-5     --num_train_epochs 3   --output_dir /opt/data/yanyu/model/transformers4/bertrdrop50_lr20_bs8_256_lv1/  --checkpointing_steps 1000

bertdblossnofl_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path /opt/data/yanyu/white_model/bert_base_uncased     --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5.py   --max_length 256   --do_train True  --report_to tensorboard    --with_tracking --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir /opt/data/yanyu/model/transformers4/bertdblossnofl_lr20_bs8_256_lv1/  --checkpointing_steps 1000

bertrdropfl_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path /opt/data/yanyu/white_model/bert_base_uncased     --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5.py   --max_length 256   --do_train True  --report_to tensorboard    --with_tracking --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir /opt/data/yanyu/model/transformers4/bertrdropfl_lr20_bs8_256_lv1/  --checkpointing_steps 1000

bertrdrop_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path /opt/data/yanyu/white_model/bert_base_uncased     --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5.py   --max_length 256   --do_train True  --report_to wandb    --with_tracking --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir /opt/data/yanyu/model/transformers4/bertrdrop_lr20_bs8_256_lv1/  --checkpointing_steps 1000

bertrdropsuperloss_lr20_bs8_256_lv1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path /opt/data/yanyu/white_model/bert_base_uncased     --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5.py --child_tune --do_train True    --learning_rate 2e-5    --max_length 256 --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/bertrdropsuperloss_lr20_bs8_256_lv1/


bertbase_lr20_bs8_256_lv12
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/bert_base_uncased    --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5_hm12.py  --do_train True  --learning_rate 2e-5  --report_to wandb   --with_tracking    --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/bertbase_lr20_bs8_256_lv12/  --checkpointing_steps 2000

berthm_lr20_bs8_256_lv12_1 1.0版本
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/bert_base_uncased    --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5_hm12.py  --do_train True  --learning_rate 2e-5  --report_to wandb   --with_tracking   --train_mode hm12   --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/berthm_lr20_bs8_256_lv12_1/  --checkpointing_steps 2000

berthm_lr20_bs8_256_lv12_1_3
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/bert_base_uncased    --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5_hm12.py  --do_train True  --learning_rate 2e-5  --report_to wandb   --with_tracking   --train_mode hm12   --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/berthm_lr20_bs8_256_lv12_1_3/  --checkpointing_steps 2000

berthm_lr20_bs8_256_lv12_1_6
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/bert_base_uncased    --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5_hm12.py  --do_train True  --learning_rate 2e-5  --report_to wandb   --with_tracking   --train_mode hm12   --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/berthm_lr20_bs8_256_lv12_1_6/  --checkpointing_steps 2000

berthm_lr20_bs8_256_lv12_1_11
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/bert_base_uncased    --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5_hm12.py  --do_train True  --learning_rate 2e-5  --report_to wandb   --with_tracking   --train_mode hm12   --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/berthm_lr20_bs8_256_lv12_1_11/  --checkpointing_steps 2000

berthm12_lr20_bs8_512_lv12_1_22
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/bert_base_uncased    --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5_hm12.py  --do_train True  --learning_rate 2e-5  --report_to wandb   --with_tracking   --train_mode hm12   --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/berthm_lr20_bs8_256_lv12_1_22/  --checkpointing_steps 2000


berthm123_lr20_bs8_256_lv123
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/bert_base_uncased    --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5_hm123.py  --do_train True  --learning_rate 2e-5  --report_to wandb   --with_tracking    --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/berthm123_lr20_bs8_256_lv123/  --checkpointing_steps 2000


sciberthm123_lr20_bs8_256_lv123 1.3
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/scibert    --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5_hm123.py  --do_train True  --learning_rate 2e-5  --report_to wandb   --with_tracking    --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/sciberthm123_lr20_bs8_256_lv123/  --checkpointing_steps 2000


berthm123_lr20_bs8_256_lv123 1.4
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/bert_base_uncased    --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5_hm123.py  --do_train True  --train_mode hm12  --learning_rate 2e-5  --report_to wandb   --with_tracking    --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/berthm123_lr20_bs8_256_lv123/  --checkpointing_steps 2000


sciberthm123_lr20_bs8_256_lv123_1_6 1.6
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/scibert    --train_file /opt/data/yanyu//data/nlpcct5/nlpcct5_hm123.py  --do_train True  --train_mode hm12  --learning_rate 2e-5  --report_to wandb   --with_tracking    --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/sciberthm123_lr20_bs8_256_lv123_1_6/  --checkpointing_steps 2000



sciberthm12_lr20_bs8_256_wos
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/scibert    --train_file /opt/data/yanyu//data/wos/wos.py  --do_train True  --train_mode hm12  --learning_rate 2e-5  --report_to wandb   --with_tracking    --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/sciberthm12_lr20_bs8_256_wos/  --checkpointing_steps 1000

bertbasehm12_lr20_bs8_256_wos
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/bert_base_uncased    --train_file /opt/data/yanyu//data/wos/wos.py  --do_train True  --train_mode hm12  --learning_rate 2e-5  --report_to wandb   --with_tracking    --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/bertbasehm12_lr20_bs8_256_wos/  --checkpointing_steps 1000

scibertmhm12_lr20_bs8_256_wos
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/scibert    --train_file /opt/data/yanyu//data/wos/wos.py  --do_train True  --train_mode hm12  --learning_rate 2e-5  --report_to wandb   --with_tracking    --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/scibertmhm12_lr20_bs8_256_wos/  --checkpointing_steps 1000


python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/model/transformers4/sciberthm12_lr20_bs8_256_wos    --train_file /opt/data/yanyu//data/wos/wos.py  --train_mode hm12  --learning_rate 2e-5    --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/sciberthm12_lr20_bs8_256_wos/  --checkpointing_steps 1000


'''


'''
computer4 test

bertbase_lr20_bs8_256_lv12
CUDA_LAUNCH_BLOCKING=1 python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /home/ydc/model/origin_model/bert_uncase/    --train_file /home/ydc/data/nlpcct5/nlpcct5_hm12.py  --do_train True  --learning_rate 2e-5  --report_to wandb   --with_tracking      --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /home/ydc/del/yy_model_set/bertbase_lr20_bs8_256_lv12  --checkpointing_steps 2000


'''



'''
gyj cp5 


berthm12_lr20_bs16_512_lv12_1_17
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/bert_nucased     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm12.py   --max_length 512   --do_train True    --report_to wandb --with_tracking  --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 4e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/berthm12_lr20_bs16_512_lv12_1_17   --checkpointing_steps 2000


berthm12_lr20_bs16_512_lv12_1_19
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/bert_nucased     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm12.py   --max_length 512   --do_train True    --report_to wandb --with_tracking  --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 4e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/berthm12_lr20_bs16_512_lv12_1_19   --checkpointing_steps 2000


berthm123_lr40_bs16_512_lv123 1.2
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/bert_nucased     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123.py   --max_length 512   --do_train True    --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/berthm123_lr40_bs16_512_lv123   --checkpointing_steps 2000
{'suset_accuracy': 0.0, 'accuracy': 0.35376355468910925, 'precision': 0.5612072329467918, 'recall': 0.4802532534008693, 'f1': 0.5098243780761337, 'micro-precision': 0.5559510788505518, 'micro-recall': 0.47663376127393314, 'micro-f1': 0.5132460665699755};threshold:0.25



sciberthm123_lr20_bs8_256_lv123 1.3
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123.py   --max_length 256   --do_train True    --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_lr20_bs8_256_lv123   --checkpointing_steps 2000


sciberthm123_lr20_bs8_256_lv123_1_5 1.5
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123.py   --max_length 256   --do_train True --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_lr20_bs8_256_lv123_1_5   --checkpointing_steps 2000

scibertmhm12_lr20_bs8_256_wos
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/scibertmhm12_lr20_bs8_256_wos   --checkpointing_steps 1000

scibertmhm12_lr20_bs8_256_wos_1_1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/scibertmhm12_lr20_bs8_256_wos_1_1   --checkpointing_steps 1000

scibertmhm12_lr20_bs8_256_wos_1_2   多任务
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/scibertmhm12_lr20_bs8_256_wos_1_2   --checkpointing_steps 1000



'''

# import torch
# if hasattr(torch.cuda, 'empty_cache'):
# 	torch.cuda.empty_cache()
# print('ok')

# import os
# os.system('''
# python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path E:/model/bert_for_test     --train_file E:/data/nlpcct5/nlpcct5.py  --do_train True  --report_to wandb   --with_tracking      --max_length 64    --per_device_eval_batch_size 32    --num_train_epochs 3     --output_dir E:/model/transformers4/bert_for_test/output_dir/  --checkpointing_steps 100
# ''')


import torch
print(torch.cuda.is_available())


'''

hm 0.5版本
不将层级关系用于loss训练
但在预测的时候，借助层级关系辅助预测

hm 0.6版本
不将层级关系用于loss训练
但在预测的时候，借助层级关系辅助预测，层级关系加权附带在原来的标签结果上


hm 1.0版本
计算loss的时候，考虑层级关系，也考虑实际标签的预测
进行预测的时候，既包含层级关系，也包含实际的预测结果

hm 1.1版本 x
计算loss的时候，考虑层级关系，也考虑实际标签的预测，（降低权重）
进行预测的时候，不考虑实际的预测结果

hm 1.2版本 x
计算loss的时候，考虑层级关系，也考虑实际标签的预测（降低权重）(重关系：原0.7+层级*0.3)
进行预测的时候，既包含层级关系，也包含实际的预测结果（降低权重）(重关系：原0.7+层级*0.3)


hm 1.3版本 x
计算loss的时候，考虑层级关系，也考虑实际标签的预测（降低权重）(重关系：原0.7+层级*0.3)
进行预测的时候，既包含层级关系，也包含实际的预测结果

hm 1.4版本 x
计算loss的时候，考虑层级关系，也考虑实际标签的预测，但基于一个loss函数进行实现，权重关系：原0.7+层级*0.3
进行预测的时候，既包含层级关系，也包含实际的预测结果权(重关系：原0.7+层级*0.3)

hm 1.5版本 
计算loss的时候，考虑层级关系，也考虑实际标签的预测，但基于一个loss函数进行实现，权重关系：原0.9+层级*0.1
进行预测的时候，既包含层级关系，也包含实际的预测结果权(重关系：原0.9+层级*0.1)

hm 1.6版本
计算loss的时候，考虑层级关系，也考虑实际标签的预测，但基于一个loss函数进行实现，权重关系：原0.9+层级*0.1
进行预测的时候，只用实际的预测结果权


hm 1.7版本
计算loss的时候，考虑实际标签的预测
进行预测的时候，既包含层级关系，也包含实际的预测结果权(重关系：原0.9+层级*0.1)

hm 1.8版本 
使用logits，而不是sigmoid
计算loss的时候，考虑层级关系，也考虑实际标签的预测，但基于一个loss函数进行实现，权重关系：原0.7+层级*0.3
进行预测的时候，既包含层级关系，也包含实际的预测结果权(重关系：原0.9+层级*0.1)

hm 1.9版本 
使用logits，而不是sigmoid
计算loss的时候，考虑层级关系，也考虑实际标签的预测，但基于一个loss函数进行实现，权重关系：原0.99+层级*0.01
进行预测的时候，既包含层级关系，也包含实际的预测结果权(重关系：原0.9+层级*0.1)


hm 1.10版本 
使用logits，而不是sigmoid,基于simcse
计算loss的时候，考虑层级关系，也考虑实际标签的预测，但基于一个loss函数进行实现，权重关系：原0.7+层级*0.3
进行预测的时候，既包含层级关系，也包含实际的预测结果权(重关系：原0.9+层级*0.1)

hm 1.11版本 x
使用logits，而不是sigmoid
计算loss的时候，考虑层级关系(双矩阵)，也考虑实际标签的预测，但基于一个loss函数进行实现，权重关系：55开
进行预测的时候，既包含层级关系，也包含实际的预测结果权(重关系：原0.9+层级*0.1)

hm 1.12版本
使用logits，而不是sigmoid
计算loss的时候，考虑层级关系(双矩阵)，也考虑实际标签的预测，但基于一个loss函数进行实现，权重关系：8 1 1
进行预测的时候，既包含层级关系，也包含实际的预测结果权 权重关系：8 1 1


hm 1.13版本
使用logits，而不是sigmoid
计算loss的时候，考虑层级关系(双矩阵)，也考虑实际标签的预测，但基于一个loss函数进行实现，权重关系：8 1 1
进行预测的时候，考虑实际的预测结果权

hm 1.14版本 x
使用logits，而不是sigmoid
计算loss的时候，考虑层级关系(双矩阵)，也考虑实际标签的预测，但基于一个loss函数进行实现，权重关系：9 0 1
进行预测的时候，考虑实际的预测结果权 8 1 1


hm 1.15版本 x
计算loss的时候，考虑层级关系(双矩阵)，也考虑实际标签的预测，但基于一个loss函数进行实现，权重关系：8 0 2
进行预测的时候，考虑实际的预测结果权 9 1 0

hm 1.16版本 微弱效果
计算loss的时候，考虑实际标签的预测，基于一个loss函数进行实现
进行预测的时候，考虑实际的预测结果权 8 1 1

hm 1.17版本 x
rdrop 计算lv12信息

hm 1.18版本 512
1.12

hm 1.20版本 512 
计算loss的时候，考虑层级关系(双矩阵)，也考虑实际标签的预测，但基于一个loss函数进行实现，使用mlp收束
进行预测的时候，考虑实际的预测结果权，使用mlp收束

hm 1.21版本 512 
lv12 原版

hm 1.22版本 256 
20 256版本

hm 1.23版本 512
lv123 

hm123 1.0 版本 256
基于hm 1.22 使用scibert

hm123 1.1版本 256
基于原版bert，直接预测lv123
{'suset_accuracy': 0.0, 'accuracy': 0.35582431254384084, 'precision': 0.5887274899120487, 'recall': 0.46449007185103075, 'f1': 0.5113932972769322, 'micro-precision': 0.5835800574849264, 'micro-recall': 0.46040270744889433, 'micro-f1': 0.5147246630958962, 'threshold': 0.27}

 
hm123 1.2版本 512
基于原版bert，直接预测lv123

hm123 1.3版本 256
基于scibert，直接预测lv123
{'suset_accuracy': 0.0, 'accuracy': 0.387794045506672, 'precision': 0.5930203048069921, 'recall': 0.5173887480773315, 'f1': 0.5449514581347026, 'micro-precision': 0.5869146514659248, 'micro-recall': 0.5143
470922203468, 'micro-f1': 0.5482399549312156, 'threshold': 0.25}


hm123 1.4版本 256
基于hm 1.22 使用bert


hm123 1.5版本 256
基于hm 1.22 使用scibert
删去S2P


hm123 1.6版本 256
基于hm 1.22 使用scibert
删去P2S

hm123 1.7版本 256
基于hm 1.22 使用scibert
删去logits

hm123 1.8版本 256
wos 基于hm 1.22，使用scibert，使用了simcse_sup训练（40epoch）

hm123 1.9版本 256
wos 基于hm 1.22，使用scibert，使用了simcse_sup训练

hm123 1.10版本 256 simcse 扩量
wos 基于hm 1.22，使用scibert，使用了simcse_sup训练


mhmwos 1.1
多任务 + s2p p2s 和logits分开


hmwos

1 使用scibert 基于1.22

1.1 加入另外一个随机logits

1.2 多任务，正式版





'''
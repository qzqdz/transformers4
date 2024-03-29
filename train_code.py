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


scibert123_lr20_bs8_256_lv123_1_10 x
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_10_simcse_base/step_5000     --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_10   --checkpointing_steps 1000 --ignore_mismatched_sizes


scibert123_lr20_bs8_256_lv123_1_11
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_10_simcse_base/step_5000     --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_11   --checkpointing_steps 1000 --ignore_mismatched_sizes
test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_11     --train_file E:/data/wos/wos.py   --max_length 256 --train_mode hm12   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_11   --checkpointing_steps 1000


scibert123_lr20_bs8_256_lv123_1_12_simcse_base 10
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/wos/wos_rand_sim_together.py   --max_length 256   --do_train True --train_mode simcse_sup   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12_simcse_base   --checkpointing_steps 1000

scibert123_lr20_bs8_256_lv123_1_12
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12_simcse_base/step_28000/     --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12   --checkpointing_steps 1000 --ignore_mismatched_sizes
test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12/     --train_file E:/data/wos/wos.py   --max_length 256   --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12   --checkpointing_steps 1000
{'suset_accuracy': 0.7853570288389912, 'accuracy': 0.8363662161682631, 'precision': 0.7012943733915713, 'recall': 0.7030939455862348, 'f1': 0.6929601918596687, 'micro-precision': 0.8618708098329254, 'micro-recall': 0.8618708098329254, 'micro-f1': 0.8618708098329254}






scibert_fl_wos  11/23 0:40
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12_simcse_base/step_28000/     --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert_fl_wos/   --checkpointing_steps 1000 --ignore_mismatched_sizes
accuracy : 0.1217764534780624  f1 :0.18266468021709056


scibert_bce_wos  11/23 7:50 # 代码正确
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12_simcse_base/step_28000/     --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert_bce_wos/   --checkpointing_steps 1000 --ignore_mismatched_sizes
acc 0.784363804050915 f1 0.8212727466212622
test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert_bce_wos     --train_file E:/data/wos/wos.py   --max_length 256 --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert_bce_wos/   --checkpointing_steps 1000
0.5962452939387255

simcse_bert_bce
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12     --train_file E:/data/wos/wos.py   --max_length 256 --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12/   --checkpointing_steps 1000
macrof1 0.6929601918596687 microf1 0.8618708098329254 acc 0.8363662161682631

scibert_cb_wos  11/23 7:58
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12_simcse_base/step_28000/     --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert_cb_wos/   --checkpointing_steps 1000 --ignore_mismatched_sizes
{'suset_accuracy': 0.8412259231669682, 'accuracy': 0.8741087581142795, 'precision': 0.8905501755879536, 'recall': 0.8905501755879536, 'f1': 0.8905501755879536, 'micro-precision': 0.8905501755879536, 'micro-recall': 0.8905501755879536, 'micro-f1': 0.8905501755879536}
test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert_cb_wos     --train_file E:/data/wos/wos.py   --max_length 256 --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert_cb_wos/   --checkpointing_steps 1000
0.7917832888921033


R-BCE-Focal
scibert_rbcefl_wos  11/23 9:40+-
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12_simcse_base/step_28000/     --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert_rbcefl_wos/   --checkpointing_steps 1000 --ignore_mismatched_sizes
{'suset_accuracy': 0.8384590826859636, 'accuracy': 0.8714838068887107, 'precision': 0.8879961689901033, 'recall': 0.8879961689901033, 'f1': 0.8879961689901033, 'micro-precision': 0.8879961689901033, 'micro-recall': 0.8879961689901033, 'micro-f1': 0.8879961689901031}

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert_rbcefl_wos     --train_file E:/data/wos/wos.py   --max_length 256 --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert_cb_wos/   --checkpointing_steps 1000
0.7908821614473451


CBloss-ntr
scibert_cbntr_wos
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12_simcse_base/step_28000/     --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert_cbntr_wos/   --checkpointing_steps 1000 --ignore_mismatched_sizes
{'suset_accuracy': 0.8385654996275407, 'accuracy': 0.8713773899471337, 'precision': 0.8877833351069491, 'recall': 0.8877833351069491, 'f1': 0.8877833351069491, 'micro-precision': 0.8877833351069491, 'micro-recall': 0.8877833351069491, 'micro-f1': 0.8877833351069491}

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert_cbntr_wos     --train_file E:/data/wos/wos.py   --max_length 256 --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert_cbntr_wos/   --checkpointing_steps 1000


DBloss
scibert_db_wos
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12_simcse_base/step_28000/     --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert_db_wos/   --checkpointing_steps 1000 --ignore_mismatched_sizes
{'suset_accuracy': 0.8417580078748537, 'accuracy': 0.8745344258805882, 'precision': 0.8909226348834735, 'recall': 0.8909226348834735, 'f1': 0.8909226
348834735, 'micro-precision': 0.8909226348834735, 'micro-recall': 0.8909226348834735, 'micro-f1': 0.8909226348834735}
test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert_db_wos     --train_file E:/data/wos/wos.py   --max_length 256 --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/scibert_db_wos/   --checkpointing_steps 1000

----------
base_scibert_bce_wos
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert      --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/base_scibert_bce_wos/   --checkpointing_steps 1000
{'suset_accuracy': 0.801319570075556, 'accuracy': 0.843035011173764, 'precision': 0.8638927317228903, 'recall': 0.8638927317228903, 'f1': 0.863892731
7228903, 'micro-precision': 0.8638927317228903, 'micro-recall': 0.8638927317228903, 'micro-f1': 0.8638927317228903}

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/base_scibert_bce_wos     --train_file E:/data/wos/wos.py   --max_length 256 --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/base_scibert_bce_wos/   --checkpointing_steps 1000
{'suset_accuracy': 0.8011067361924018, 'accuracy': 0.8428931219183278, 'precision': 0.7214710990171631, 'recall': 0.7257800624118603, 'f1': 0.7165594987987893, 'micro-precision': 0.8637863147813132, 'micro
-recall': 0.8637863147813132, 'micro-f1': 0.8637863147813132}


----------
base_bert_bce_wos
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert      --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12  --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/base_bert_bce_wos/   --checkpointing_steps 1000
{'suset_accuracy': 0.7194849420027668, 'accuracy': 0.7856762796637022, 'precision': 0.6352143049531673, 'recall': 0.6427698284552018, 'f1': 0.6275248
178151367, 'micro-precision': 0.8187719484942003, 'micro-recall': 0.8187719484942003, 'micro-f1': 0.8187719484942003}
test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/base_bert_bce_wos     --train_file E:/data/wos/wos.py   --max_length 256 --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/base_bert_bce_wos/   --checkpointing_steps 1000


----------
bhifn_scibert_bce_wos
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert      --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/bhifn_scibert_bce_wos/   --checkpointing_steps 1000
{'suset_accuracy': 0.7639672235819942, 'accuracy': 0.8196942286545155, 'precision': 0.8475577311908056, 'recall': 0.8475577311908056, 'f1': 0.8475577311908056, 'micro-precision': 0.8475577311908056, 'micro
-recall': 0.8475577311908056, 'micro-f1': 0.8475577311908056}

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/bhifn_scibert_bce_wos     --train_file E:/data/wos/wos.py   --max_length 256 --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/bhifn_scibert_bce_wos/   --checkpointing_steps 1000
 0.6636446733145177

----------
base_scibert_cbntr_wos
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert      --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/base_scibert_cbntr_wos/   --checkpointing_steps 1000

macro-f1 0.7965687381137191

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/base_scibert_cbntr_wos     --train_file E:/data/wos/wos.py   --max_length 256 --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/base_scibert_cbntr_wos/   --checkpointing_steps 1000
{'suset_accuracy': 0.8385654996275407, 'accuracy': 0.8718385300273013, 'precision': 0.797107663291949, 'recall': 0.7994214272216097, 'f1': 0.7965687381137191, 'micro-precision': 0.8884750452272002, 'micro-
recall': 0.8884750452272002, 'micro-f1': 0.8884750452272002}

ma 0.7965687381137191
mi 0.8884750452272002
acc 0.8718385300273013


----------
bhifn_scibert_cbntr_wos
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12_simcse_base/step_28000      --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/bhifn_scibert_cbntr_wos/   --checkpointing_steps 1000 --ignore_mismatched_sizes
{'suset_accuracy': 0.8447376822390125, 'accuracy': 0.8777624064417601, 'precision': 0.894274768543152, 'recall': 0.894274768543152, 'f1': 0.894274768
543152, 'micro-precision': 0.8103550539175588, 'micro-recall': 0.8015228470736716, 'micro-f1': 0.8029892466106487}

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/bhifn_scibert_cbntr_wos     --train_file E:/data/wos/wos.py   --max_length 256 --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/bhifn_scibert_cbntr_wos/   --checkpointing_steps 1000
{'suset_accuracy': 0.8437799297648185, 'accuracy': 0.8773722109893104, 'precision': 0.8097603960870913, 'recall': 0.8009405040902832, 'f1': 0.8024876031286343, 'micro-precision': 0.8941683516015749, 'micro
-recall': 0.8941683516015749, 'micro-f1': 0.8941683516015749}

----------
bhifn_scibert_cbntr_wos_no_simcse
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert      --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/bhifn_scibert_cbntr_wos_no_simcse/   --checkpointing_steps 1000
{'suset_accuracy': 0.8366499946791529, 'accuracy': 0.8719449469688784, 'precision': 0.7986816009913603, 'recall': 0.7894680603841501, 'f1': 0.7895745486024193, 'micro-precision': 0.8895924231137597, 'micro-recall': 0.8895924231137597, 'micro-f1': 0.8895924231137597}


----------
base_scibert_bce_wos_simcse
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12_simcse_base/step_28000      --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/base_scibert_cbntr_wos_simcse/   --checkpointing_steps 1000 --ignore_mismatched_sizes
{'suset_accuracy': 0.8352665744386506, 'accuracy': 0.8693554680571682, 'precision': 0.7911386708188566, 'recall': 0.7746665669811709, 'f1': 0.7756939492209841, 'micro-precision': 0.8863999148664468, 'micro-recall': 0.8863999148664468, 'micro-f1': 0.8863999148664468}


----------
base_scibert_bce_wos_simcse_1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12_simcse_base/step_28000      --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/base_scibert_cbntr_wos_simcse_1/   --checkpointing_steps 1000 --ignore_mismatched_sizes
{'suset_accuracy': 0.8332446525486857, 'accuracy': 0.8670852399701902, 'precision': 0.7746504625057323, 'recall': 0.768148951930167, 'f1': 0.7665451562389283, 'micro-precision': 0.884005533680962, 'micro-recall': 0.884005533680962, 'micro-f1': 0.884005533680962}


----------
base_scibert_cbntr_wos_simcse
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/scibert123_lr20_bs8_256_lv123_1_12_simcse_base/step_28000      --train_file E:/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/base_scibert_cbntr_wos_simcse/   --checkpointing_steps 1000 --ignore_mismatched_sizes
{'suset_accuracy': 0.8376077471533468, 'accuracy': 0.8710936114362613, 'precision': 0.8051423569768571, 'recall': 0.7985534905278445, 'f1': 0.7995508
628972241, 'micro-precision': 0.8878365435777376, 'micro-recall': 0.8878365435777376, 'micro-f1': 0.8878365435777376}

----------
hm12 debug
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm12.py   --max_length 256   --do_train True  --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm12_lr20_bs8_256_lv12_test   --checkpointing_steps 2

python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/bert     --train_file E:/data/nlpcct5/nlpcct5_hm123.py   --max_length 256   --do_train True  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/berthm123_lr20_bs8_512_lv123   --checkpointing_steps 2




hm123 test
sciberthm123_lr20_bs8_256_lv123_1 s2p p2s logits
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path  E:/model/transformers4/sciberthm123_lr20_bs8_256_lv123_1     --train_file E:/data/nlpcct5/nlpcct5_hm123_f.py   --max_length 256  --train_mode hm12 --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_lr20_bs8_512_lv123_1/   --checkpointing_steps 2000
{'suset_accuracy': 0.0394, 'accuracy': 0.7265133007429435, 'precision': 0.9068068980159847, 'recall': 0.7828571773857377, 'f1': 0.8337299328720417, 'micro-precision': 0.9020995471564957, 'micro-recall': 0.
7832947504595901, 'micro-f1': 0.8385098260734883, 'threshold': 0.38}

python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path  E:/model/transformers4/sciberthm123_lr20_bs8_256_lv123_1     --train_file E:/data/nlpcct5/nlpcct5_hm123_t.py  --threshold 0.38 --max_length 256  --train_mode hm12 --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_lr20_bs8_256_lv123_1/   --checkpointing_steps 2000


----------
hm123
划分val、test、train后的补充实验

----------
sciberthm123_256_1 1 logits p2s s2p
本机跑的
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_1   --checkpointing_steps 2000

val
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/sciberthm123_256_1     --train_file E:/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --train_mode hm12   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_1   --checkpointing_steps 2000
{'suset_accuracy': 0.0336, 'accuracy': 0.7183688402935069, 'precision': 0.9003829029269664, 'recall': 0.7781086333764275, 'f1': 0.8281805041521522, 'micro-precision': 0.8949087566256577, 'micro-recall': 0.
7788180023149724, 'micro-f1': 0.8328373151308306, 'threshold': 0.37}


test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/sciberthm123_256_1     --train_file E:/data/nlpcct5/nlpcct5_hm123.py   --max_length 256  --threshold 0.37  --train_mode hm12   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_1   --checkpointing_steps 2000
{'suset_accuracy': 0.0188, 'accuracy': 0.6652984807182579, 'precision': 0.8743459492872109, 'recall': 0.7325835363955182, 'f1': 0.7895888909189536, 'micro-precision': 0.8673531193216233, 'micro-recall': 0.7324433532811621, 'micro-f1': 0.7942098111533236, 'threshold': 0.37}



----------

sciberthm123_256_2 2 p2s s2p
在s4服务器上跑的
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_2   --checkpointing_steps 2000

val threshold
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_2/     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256    --train_mode hm12    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_2/   --checkpointing_steps 2000
{'suset_accuracy': 0.0136, 'accuracy': 0.6723988354249777, 'precision': 0.9048924867779279, 'recall': 0.7233972285033508, 'f1': 0.7957105976707872, 'micro-precision': 0.8984526519942659, 'micro-recall': 0.725
454483556887, 'micro-f1': 0.8027386423566639, 'threshold': 0.36000000000000004}

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_2/     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123.py --threshold 0.36000000000000004    --max_length 256    --train_mode hm12    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_2/   --checkpointing_steps 2000
{'suset_accuracy': 0.0062, 'accuracy': 0.6352641324121308, 'precision': 0.8917970488835434, 'recall': 0.688989428282835, 'f1': 0.7677042196394966, 'micro-precision': 0.8829852178083983, 'micro-recall': 0.6904847151893339, 'micro-f1': 0.7749595767276763, 'threshold': 0.36000000000000004}



----------
sciberthm123_256_3 3 logits s2p
本机跑的
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_3   --checkpointing_steps 2000
{'suset_accuracy': 0.0028, 'accuracy': 0.50723329695319, 'precision': 0.6945154609992691, 'recall': 0.6411912119545401, 'f1': 0.6585797548310046, 'micro-precision': 0.6892209845226278, 'micro-recall': 0.6382344930891264, 'micro-f1': 0.6627485638532921};threshold:0.30000000000000004

死机后重新跑 x
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/sciberthm123_256_3/step_30000     --train_file E:/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_3/step_30000/more   --checkpointing_steps 2000

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/sciberthm123_256_3     --train_file E:/data/nlpcct5/nlpcct5_hm123.py   --max_length 256  --threshold 0.30000000000000004  --train_mode hm12   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_3/   --checkpointing_steps 2000
{'suset_accuracy': 0.001, 'accuracy': 0.4514209731616783, 'precision': 0.6427268984687483, 'recall': 0.5895690070440982, 'f1': 0.6063105975954075, 'micro-precision': 0.6351868631765816, 'micro-recall': 0.5856307435254804, 'micro-f1': 0.6094029983145569, 'threshold': 0.30000000000000004}

----------
sciberthm123_256_4 4 logits p2s
电脑2跑的 x
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/scibert     --train_file D:/data/nlpcct5/nlpcct5_hm123_t.py --train_mode hm12 --max_length 256   --do_train True  --report_to wandb    --with_tracking --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/sciberthm123_256_4/ --checkpointing_steps 2000

在s4服务器上跑的
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_4   --checkpointing_steps 2000

val
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_4     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256  --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_4   --checkpointing_steps 2000
{'suset_accuracy': 0.0126, 'accuracy': 0.6654546052329331, 'precision': 0.8960462810443769, 'recall': 0.7204594668609684, 'f1': 0.7902986301429514, 'micro-precision': 0.8880545503984606, 'micro-recall': 0.722
696942874651, 'micro-f1': 0.796888050526948, 'threshold': 0.37}



test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_4     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123.py   --max_length 256 --threshold 0.37 --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_4   --checkpointing_steps 2000
{'suset_accuracy': 0.0046, 'accuracy': 0.6239927012323723, 'precision': 0.8792323448909312, 'recall': 0.6820690685075075, 'f1': 0.7589278662644918, 'micro-precision': 0.8696547835892098, 'micro-recall': 0.6837672412323326, 'micro-f1': 0.7655890578319923, 'threshold': 0.37}


----------
bertbasehm123_256_5 5 logits p2s s2p
在s3服务器上跑的 x
python ./examples/pytorch/text-classification/run_glue_no_trainer.py    --model_name_or_path  /opt/data/yanyu/white_model/bert_base_uncased    --train_file /opt/data/yanyu/data/nlpcct5/nlpcct5_hm123_t.py  --do_train True  --train_mode hm12  --learning_rate 2e-5  --report_to wandb   --with_tracking    --max_length 256    --per_device_train_batch_size 8   --per_device_eval_batch_size 8    --num_train_epochs 3     --output_dir /opt/data/yanyu/model/transformers4/bertbasehm123_256_6/  --checkpointing_steps 2000

在s4服务器上跑的
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/bert_uncased     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True  --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/bertbasehm123_256_5   --checkpointing_steps 2000

val
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/transformers4/bertbasehm123_256_5/     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123_t.py     --max_length 256    --train_mode hm12    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/bertbasehm123_256_5/   --checkpointing_steps 2000
{'suset_accuracy': 0.0226, 'accuracy': 0.6974232828962645, 'precision': 0.9027273543235231, 'recall': 0.7527673760432894, 'f1': 0.813695608436024, 'micro-precision': 0.896752655538695, 'micro-recall': 0.75444
27044324913, 'micro-f1': 0.8194651160640826, 'threshold': 0.37}

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/transformers4/bertbasehm123_256_5     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123.py --threshold 0.37    --max_length 256    --train_mode hm12    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/bertbasehm123_256_5/   --checkpointing_steps 2000
{'suset_accuracy': 0.0122, 'accuracy': 0.6506680674287832, 'precision': 0.8829703661805033, 'recall': 0.7107795541854631, 'f1': 0.7793164881091419, 'micro-precision': 0.8753563044936284, 'micro-recall': 0.7120692888684296, 'micro-f1': 0.7853147181873736, 'threshold': 0.37}

----------
bertbasehm123_256_6 6 no mlp 0.4 0.3 0.3
在s4服务器上跑的
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True  --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/bertbasehm123_256_6   --checkpointing_steps 2000
{'suset_accuracy': 0.0, 'accuracy': 0.05236197780547644, 'precision': 0.608, 'recall': 0.05236197780547625, 'f1': 0.09610261206114148, 'micro-precision'
: 0.608, 'micro-recall': 0.0517464424320828, 'micro-f1': 0.09537554119344921};threshold:0.2

----------
sciberthm123_256_7 7 no mlp 0.98 0.01 0.01
在s4服务器上跑的 x
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True  --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_7   --checkpointing_steps 2000

在本机上跑的
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_7   --checkpointing_steps 2000
{'suset_accuracy': 0.0, 'accuracy': 0.4148957189188589, 'precision': 0.6482174390968508, 'recall': 0.5261266262959607, 'f1': 0.573043303781627, 'micro-precision': 0.6426631002633009, 'micro-recall': 0.5234901613671955, 'micro-f1': 0.5769872985497458};threshold:0.27

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/sciberthm123_256_7     --train_file E:/data/nlpcct5/nlpcct5_hm123.py   --max_length 256  --threshold 0.27  --train_mode hm12   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_7/   --checkpointing_steps 2000
{'suset_accuracy': 0.0, 'accuracy': 0.3802734576206964, 'precision': 0.6125031116432587, 'recall': 0.49135521734765525, 'f1': 0.5372214733855412, 'micro-precision': 0.605720814115855, 'micro-recall': 0.48812507459124, 'micro-f1': 0.5406017806059348, 'threshold': 0.27}


----------
sciberthm123_256_8 8 no mlp 0.4 0.3 0.3
在s4服务器上跑的
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True  --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/bertbasehm123_256_8   --checkpointing_steps 2000
{'suset_accuracy': 0.0, 'accuracy': 0.23486406042047125, 'precision': 0.42539391663891657, 'recall': 0.34000634444245126, 'f1': 0.37129483694812715, 'micro-precision': 0.42541483781179745, 'micro-recall': 0.33820725811942537, 'micro-f1': 0.3768313845978778};threshold:0.2


test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/transformers4/bertbasehm123_256_8/     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123.py --threshold 0.2    --max_length 256    --train_mode hm12    --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/bertbasehm123_256_8/   --checkpointing_steps 2000
{'suset_accuracy': 0.0, 'accuracy': 0.2199126586864963, 'precision': 0.4032839022089022, 'recall': 0.3211356033210583, 'f1': 0.3511557621572125, 'micro-precision': 0.4038018557189296, 'micro-recall': 0.3197960888616098, 'micro-f1': 0.3569226670726138, 'threshold': 0.2}



----------
sciberthm123_256_9
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/scibert     --train_file D:/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True  --report_to wandb --with_tracking   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/sciberthm123_256_9/   --checkpointing_steps 2000
{"suset_accuracy": 0.0, "accuracy": 0.4127715824457866, "precision": 0.6359777743498332, "recall": 0.5315549054677693, "f1": 0.5711764773670871, "micro-precision": 0.6306364836057253, "micro-recall": 0.5287328930346565, "micro-f1": 0.5752062442709925, "threshold": 0.26}
在电脑2跑的

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/transformers4/sciberthm123_256_9     --train_file D:/data/nlpcct5/nlpcct5_hm123.py  --threshold 0.26 --max_length 256   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/sciberthm123_256_9/   --checkpointing_steps 2000
{'suset_accuracy': 0.0, 'accuracy': 0.3802223847445191, 'precision': 0.6015560456863398, 'recall': 0.49844082458728645, 'f1': 0.5372817834924953, 'micro-precision': 0.5959913014935172, 'micro-recall': 0.4953028830579851, 'micro-f1': 0.5410020764080934, 'threshold': 0.26}
----------
bertbasehm123_256_9
在电脑2跑的
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/white_model/bert_base_uncased     --train_file D:/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True  --report_to wandb --with_tracking   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/bertbasehm123_256_9/   --checkpointing_steps 2000
{'suset_accuracy': 0.0, 'accuracy': 0.36475755243311003, 'precision': 0.5602629901986558, 'recall': 0.5022958774292443, 'f1': 0.5225951493332519, 'micro-precision': 0.555814349826734, 'micro-recall': 0.4996255191666099, 'micro-f1': 0.5262242620364478, 'threshold': 0.23}

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path D:/model/transformers4/bertbasehm123_256_9     --train_file D:/data/nlpcct5/nlpcct5_hm123.py  --threshold 0.26 --max_length 256   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir D:/model/transformers4/bertbasehm123_256_9/   --checkpointing_steps 2000
{'suset_accuracy': 0.0, 'accuracy': 0.34068217785584975, 'precision': 0.5674677853355794, 'recall': 0.45173657807928685, 'f1': 0.4954540301812979, 'micro-precision': 0.5628117040176374, 'micro-recall': 0.4482976147852625, 'micro-f1': 0.4990699616596439, 'threshold': 0.26}

----------
sciberthm123_256_10 simcse bce

python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/transformers4/simcse_nlpcc_scibert/step_10000     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True  --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_10/   --checkpointing_steps 2000 --ignore_mismatched_sizes
{'suset_accuracy': 0.0232, 'accuracy': 0.6862124274814689, 'precision': 0.8839535702756137, 'recall': 0.7516608721223736, 'f1': 0.8051999145177897, 'micro-precision': 0.8781423262653762, 'micro-recall': 0.7521787975760877, 'micro-f1': 0.8102944007921591};threshold:0.34

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_10     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_hm123.py  --threshold 0.34  --max_length 256  --train_mode hm12 --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/sciberthm123_256_10/   --checkpointing_steps 2000
{'suset_accuracy': 0.0118, 'accuracy': 0.637507490843347, 'precision': 0.8617760862580831, 'recall': 0.7076695225112466, 'f1': 0.7687605602890942, 'micro-precision': 0.8540934048985132, 'micro-recall': 0.7080967725436039, 'micro-f1': 0.7742729306487695, 'threshold': 0.34}


----------
sciberthm123_256_11 cb
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_11/   --checkpointing_steps 2000
{'suset_accuracy': 0.0682, 'accuracy': 0.7677470786057634, 'precision': 0.9177819916272237, 'recall': 0.8218072173784249, 'f1': 0.8617067707167338, 'micro-precision': 0.9135954290903587, 'micro-recall': 0.8219684074351468, 'micro-f1': 0.8653632486290814};threshold:0.5800000000000001

test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/sciberthm123_256_11     --train_file E:/data/nlpcct5/nlpcct5_hm123.py  --threshold 0.58  --max_length 256  --train_mode hm12 --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_11/   --checkpointing_steps 2000
{'suset_accuracy': 0.0412, 'accuracy': 0.7142526126864944, 'precision': 0.8904032953819485, 'recall': 0.7795319989317068, 'f1': 0.8246803925710505, 'micro-precision': 0.8850376624131053, 'micro-recall': 0.7792610778647299, 'micro-f1': 0.8287879886849931, 'threshold': 0.58}

----------
补充实验
sciberthm123_256_13 iso + cbntr
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/nlpcct5/iso_nlpcct5_hm123_t.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_13/   --checkpointing_steps 2000
{'suset_accuracy': 0.0654, 'accuracy': 0.7649810657980244, 'precision': 0.4919094837494828, 'recall': 0.5418004391798211, 'f1': 0.49550806541325054,
'micro-precision': 0.8559825512313427, 'micro-recall': 0.8678625651858208, 'micro-f1': 0.8618816221841633};threshold:0.52


----------
补充实验
sciberthm123_256_14 iso + bce
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/nlpcct5/iso_nlpcct5_hm123_t.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_14/   --checkpointing_steps 2000


----------
simcse_nlpcc_scibert
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/nlpcct5/nlpcct5_rand_sim_together.py   --max_length 256   --do_train True  --train_mode simcse_sup  --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/simcse_nlpcc_scibert   --checkpointing_steps 2000
<2000:2.7321252822875977>
<4000:4.538190841674805>
<2000:1.7465476989746094>
<4000:3.7531967163085938>
<2000:1.5785503387451172>
<4000:3.169832706451416>

----------
sciberthm123_256_12 cb-ntr
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/nlpcct5/nlpcct5_hm123_t.py   --max_length 256   --do_train True --train_mode hm12 --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_12/   --checkpointing_steps 2000
{'suset_accuracy': 0.0746, 'accuracy': 0.7721063206263992, 'precision': 0.9188324266445629, 'recall': 0.8261188834041775, 'f1': 0.86461936879495, 'micro-precision': 0.9144686058552198, 'micro-recall': 0.8267855927010281, 'micro-f1': 0.8684194058804073};threshold:0.5800000000000001


test
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/transformers4/sciberthm123_256_12     --train_file E:/data/nlpcct5/nlpcct5_hm123.py  --threshold 0.58  --max_length 256  --train_mode hm12 --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/sciberthm123_256_12/   --checkpointing_steps 2000
{'suset_accuracy': 0.0358, 'accuracy': 0.7167122149889884, 'precision': 0.8925432064822003, 'recall': 0.7809790402512193, 'f1': 0.8265299099741261, 'micro-precision': 0.8872276556599206, 'micro-recall': 0.7810683170511312, 'micro-f1': 0.8307703467285652, 'threshold': 0.58}




----------
以下代码无效！！
----------
wos_scibert_simcse_d1_r10_bs16
在s4服务器上跑的
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/wos_new/wos_rand_sim_together.py   --max_length 256   --do_train True  --train_mode simcse_sup  --per_device_eval_batch_size 16  --per_device_train_batch_size 16  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/wos_scibert_simcse_d1_r10   --checkpointing_steps 2000

C:/Users/Administrator/Desktop/yindechun/model/transformers4/wos_scibert_simcse_d1_r10\step_22000
0.02


step24000
0.46

wos_simcse_1_1
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/transformers4/wos_scibert_simcse_d1_r10/step_22000     --train_file C:/Users/Administrator/Desktop/yindechun/data/wos_new/wos.py   --max_length 256   --do_train True --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/wos_simcse_1_1/   --checkpointing_steps 1000

wos_scibert_2
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/wos_new/wos.py   --max_length 256   --do_train True --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/wos_scibert_2/   --checkpointing_steps 1000


test to run
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/wos_new/wos.py   --max_length 256   --do_train True --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/wos_simcse_1_1_ttr/   --checkpointing_steps 1000

----------
wos_scibert_simcse_d1_r10_bs_8
在本机跑的
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/wos_new/wos_rand_sim_together.py   --max_length 256   --do_train True --train_mode simcse_sup   --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/wos_scibert_simcse_d1_r10_bs_8   --checkpointing_steps 1000

test run
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/wos_new/wos.py   --max_length 256   --do_train True --train_mode hm12  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/simcse_test   --checkpointing_steps 5

----------
wos_scibert_simcse_d1_r1_bs_8




----------
wos_scibert_1 原版 logits p2s s2p

python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/wos_new/nlpcct5_hm123_t.py   --max_length 256   --do_train True  --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/bertbasehm123_256_6   --checkpointing_steps 2000



python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path E:/model/white_model/scibert     --train_file E:/data/wos_new/wos.py   --max_length 256   --do_train True  --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir E:/model/transformers4/wos_scibert_1   --checkpointing_steps 1000



----------
simcse_wos_scibert_1 原版 logits p2s s2p

----------

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



scibertmhm12_lr20_bs8_256_wos_1_3 rdrop
python ./examples/pytorch/text-classification/run_glue_no_trainer.py     --model_name_or_path C:/Users/Administrator/Desktop/yindechun/model/blank_model/scibert     --train_file C:/Users/Administrator/Desktop/yindechun/data/wos/wos.py   --max_length 256   --do_train True --train_mode hm12   --report_to wandb --with_tracking  --per_device_eval_batch_size 8  --per_device_train_batch_size 8  --learning_rate 2e-5     --num_train_epochs 3   --output_dir C:/Users/Administrator/Desktop/yindechun/model/transformers4/scibertmhm12_lr20_bs8_256_wos_1_3   --checkpointing_steps 1000



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
python train.py --lr 0.005 --momentum 0.5 --num_hidden 2 --sizes 300,300 --activation relu --loss ce --opt adam --batch_size 20 --epochs 5 --anneal true  --save_dir ../save_dir/ --expt_dir ../expt_dir/ --train ./train.csv --val ./valid.csv --test ./test.csv --pretrain False --state 0 --testing false

# python train.py --test ../save_dir/best/test_1.csv --save_dir ../save_dir/best/ --expt_dir ../expt_dir/ --pretrain true --state 1 --testing true

# python train.py --test ../save_dir/best/test_2.csv --save_dir ../save_dir/best/ --expt_dir ../expt_dir/ --pretrain true --state 2 --testing true



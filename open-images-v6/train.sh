python main.py \
--train \
--train_annot_fp "./small-train-annotations-bbox-target.csv" \
--eval \
--val_annot_fp "./validation-annotations-bbox-target.csv" \
--model "faster_rcnn_X_101_32x8d_FPN_3x" \
--max_iter 300 \
--lr 5e-4 \
--gamma 0.5 \
--lr_decay_steps 250 300 \
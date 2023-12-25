seed=1024
repeat=5

# cgl model
method=cgm  # Condensed Graph Memory (CGM)

# training
hidden=512
layers_num=3
epoch=500
lr=0.001

for dataset in corafull arxiv reddit products; do

    # Set budget based on the dataset
    if [ "$dataset" = "corafull" ]; then
        budget=(4)
    elif [ "$dataset" = "arxiv" ]; then
        budget=(29)
    elif [ "$dataset" = "reddit" ]; then
        budget=(40)
    elif [ "$dataset" = "products" ]; then
        budget=(318)
    fi

    # Loop over each budget value
    for b in "${budget[@]}"; do
        python train.py \
        --data-dir /scratch/user/uqyliu71/PUMA_data/data \
        --result-path /scratch/user/uqyliu71/PUMA_data/results \
        --seed $seed \
        --repeat $repeat \
        --cls-epoch $epoch \
        --cgl-method $method \
        --dataset-name $dataset \
        --budget $b \
        --hidden $hidden \
        --layers-num $layers_num \
        --lr $lr \
        --pseudo-label \
        --evaluate \
        --tim \
        --edge-free \
        --retrain \
        --cgm-args "{'n_encoders': 100, 'update_epoch': 10, 'feat_init': 'randomChoice', 'feat_lr': 1e-3, 'hid_dim': 4096, 'emb_dim': 256, 'n_layers': 2, 'hop': 1, 'activation': True, 'otp': True}";
    done
done
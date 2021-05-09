mode=$1
dataset=$2
model=$3
CUDA=$4

# try catch
if [ ! $model ]; then
    model='none'
    CUDA=0
fi

# running mode (hierarchical/graph):
# no-hierarchical: Seq2Seq; hierarchical: HRED, VHRED, WSeq, ...
# graph: MTGAT, MTGCN; no-graph: Seq2Seq, HRED, VHRED, WSeq, ...
if [ $model = 'HRED' ]; then
    hierarchical=1
    graph=0
elif [ $model = 'Seq2Seq' ]; then
    hierarchical=0
    graph=0
elif [ $model = 'Seq2Seq_MHA' ]; then
    hierarchical=0
    graph=0
else
    hierarchical=0
    graph=0
fi

# maxlen and batch_size
# for dailydialog dataset, 20 and 150 is the most appropriate settings
if [ $hierarchical = 1 ]; then
    maxlen=50
    tgtmaxlen=30
    batch_size=128
elif [ $transformer_decode = 1 ]; then
    maxlen=200
    tgtmaxlen=25
    batch_size=64
else
    maxlen=150
    tgtmaxlen=25
    batch_size=64
fi

# ========== Ready Perfectly ========== #
echo "========== $mode begin =========="

if [ $mode = 'lm' ]; then
    echo "[!] Begin to train the N-gram Language Model"
    python utils.py \
        --dataset $dataset \
        --mode lm 

elif [ $mode = 'vocab' ]; then
    # Generate the src and tgt vocabulary
    echo "[!] Begin to generate the vocab"
    
    if [ ! -d "./processed/$dataset" ]; then
        mkdir -p ./processed/$dataset
        echo "[!] cannot find the folder, create ./processed/$dataset"
    else
        echo "[!] ./processed/$dataset: already exists"
    fi
    
    python utils.py \
        --mode vocab \
        --cutoff 50000 \
        --field src \
        --vocab ./processed/$dataset/iptvocab.pkl \
        --file ./dataset/$dataset/train.json

    python utils.py \
        --mode vocab \
        --cutoff 50000 \
        --field trg \
        --vocab ./processed/$dataset/optvocab.pkl \
        --file ./dataset/$dataset/train.json
        
    # generate the whole vocab for VHRED and KgCVAE (Variational model)
    python utils.py \
        --mode vocab \
        --cutoff 50000 \
        --field all \
        --vocab ./processed/$dataset/vocab.pkl \
        --file ./dataset/$dataset/train.json

elif [ $mode = 'train' ]; then
    # cp -r ./ckpt/$dataset/$model ./bak/ckpt    # too big, stop back up it
    rm -rf ./ckpt/$dataset/$model
    mkdir -p ./ckpt/$dataset/$model
    
    # create the training folder
    if [ ! -d "./processed/$dataset/$model" ]; then
        mkdir -p ./processed/$dataset/$model
    else
        echo "[!] ./processed/$dataset/$model: already exists"
    fi
    
    # delete traninglog.txt
    if [ ! -f "./processed/$dataset/$model/trainlog.txt" ]; then
        echo "[!] ./processed/$dataset/$model/trainlog.txt doesn't exist"
    else
        rm ./processed/$dataset/$model/trainlog.txt
    fi
    
    # delete metadata.txt
    if [ ! -f "./processed/$dataset/$model/metadata.txt" ]; then
        echo "[!] ./processed/$dataset/$model/metadata.txt doesn't exist"
    else
        rm ./processed/$dataset/$model/metadata.txt
    fi
    
    cp -r tblogs/$dataset/ ./bak/tblogs
    rm tblogs/$dataset/$model/*
    
    # Because of the posterior, the Variational models need to bind the src and tgt vocabulary
    if [[ $model = 'VHRED' || $model = 'KgCVAE' ]]; then
        echo "[!] VHRED or KgCVAE, src vocab == tgt vocab"
        src_vocab="./processed/$dataset/vocab.pkl"
        tgt_vocab="./processed/$dataset/vocab.pkl"
    else
        src_vocab="./processed/$dataset/iptvocab.pkl"
        tgt_vocab="./processed/$dataset/optvocab.pkl"
    fi
    
    # dropout for transformer
    if [ $model = 'Transformer' ]; then
        # other repo set the 0.1 as the dropout ratio, remain it
        dropout=0.3
        lr=1e-4
        lr_mini=1e-6
    else
        dropout=0.3
        lr=1e-4
        lr_mini=1e-6
    fi
    
    echo "[!] back up finished"
    
    # Train
    echo "[!] Begin to train the model"
    
    # set the lr_gamma as 1, means that don't use the learning rate schedule
    # Transformer: lr(threshold) 1e-4, 1e-6 / others: lr(threshold) 1e-4, 1e-6
    CUDA_VISIBLE_DEVICES="$CUDA" python train.py \
        --src_train ./data/$dataset/src-train.txt \
        --tgt_train ./data/$dataset/tgt-train.txt \
        --src_test ./data/$dataset/src-test.txt \
        --tgt_test ./data/$dataset/tgt-test.txt \
        --src_dev ./data/$dataset/src-dev.txt \
        --tgt_dev ./data/$dataset/tgt-dev.txt \
        --src_vocab $src_vocab \
        --tgt_vocab $tgt_vocab \
        --train_graph ./processed/$dataset/train-graph.pkl \
        --test_graph ./processed/$dataset/test-graph.pkl \
        --dev_graph ./processed/$dataset/dev-graph.pkl \
        --pred ./processed/${dataset}/${model}/pure-pred.txt \
        --min_threshold 0 \
        --max_threshold 100 \
        --seed 30 \
        --epochs 100 \
        --lr $lr \
        --batch_size $batch_size \
        --model $model \
        --utter_n_layer 2 \
        --utter_hidden 512 \
        --teach_force 1 \
        --context_hidden 512 \
        --decoder_hidden 512 \
        --embed_size 256 \
        --patience 5 \
        --dataset $dataset \
        --grad_clip 3.0 \
        --dropout $dropout \
        --d_model 512 \
        --nhead 4 \
        --num_encoder_layers 8 \
        --num_decoder_layers 8 \
        --dim_feedforward 2048 \
        --hierarchical $hierarchical \
        --transformer_decode $transformer_decode \
        --graph $graph \
        --maxlen $maxlen \
        --tgt_maxlen $tgtmaxlen \
        --position_embed_size 30 \
        --context_threshold 2 \
        --dynamic_tfr 15 \
        --dynamic_tfr_weight 0.0 \
        --dynamic_tfr_counter 10 \
        --dynamic_tfr_threshold 1.0 \
        --bleu nltk \
        --contextrnn \
        --no-debug \
        --lr_mini $lr_mini \
        --lr_gamma 0.5 \
        --warmup_step 4000 \
        --gat_heads 8 \

fi
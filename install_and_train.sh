pip install -r requirements.txt
python setup.py develop

ln -s /root/data_atd/ImageNet ./


python -m torch.distributed.launch \
    --use-env \
    --nproc_per_node=4 \
    --master_port=1145  \
    basicsr/train.py \
    -opt options/train/EDSR_CT_SRx4.yml \
    --launcher pytorch

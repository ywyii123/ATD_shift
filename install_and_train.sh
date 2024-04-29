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

# current_date=$(date +%Y%m%d)
# rundir_logs="${current_date}_tb_loggers"
# rundir_experiments="${current_date}_experiments"

cp -rf experiments/ /root/data_atd/experiments
cp -rf tb_loggers/ /root/data_atd/tb_loggers

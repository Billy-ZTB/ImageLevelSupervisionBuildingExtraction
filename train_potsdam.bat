set DATA_ROOT=C:\ZTB\Dataset\VOC_potsdam
set TRAIN_LIST=C:\ZTB\Dataset\VOC_potsdam\train.txt
set CAM_WEIGHTS=sess/res50_cam_potsdam.pth
set CAM_OUT=result/cam_adv_mask_potsdam
set IR_LABEL_OUT=result/ir_label_potsdam
set IRN_WEIGHTS=sess/res50_irn_potsdam.pth
set SEM_SEG_OUT=result/sem_seg_potsdam

::python run_sample.py --train_cam_pass True ^
::    --cam_batch_size 32 ^
::    --cam_num_epoches 60 ^
::    --voc12_root %DATA_ROOT% ^
::    --train_list %TRAIN_LIST% ^
::    --cam_learning_rate 0.001 ^
::    --cam_weights_name %CAM_WEIGHTS%

::python obtain_CAM_masking.py --adv_iter 2 --AD_coeff 7 --AD_stepsize 0.08 --score_th 0.6 ^
::    --voc12_root %DATA_ROOT% ^
::    --train_list %TRAIN_LIST% ^
::    --cam_weights_name %CAM_WEIGHTS% ^
::    --cam_out_dir %CAM_OUT%

python run_sample.py --eval_cam_pass True --cam_out_dir %CAM_OUT% --voc12_root %DATA_ROOT% --infer_list %TRAIN_LIST%

python run_sample.py --cam_to_ir_label_pass True --conf_fg_thres 0.5 --conf_bg_thres 0.4 ^
    --cam_out_dir %CAM_OUT% ^
    --voc12_root %DATA_ROOT% ^
    --train_list %TRAIN_LIST% ^
    --ir_label_out_dir %IR_LABEL_OUT%

python run_sample.py --train_irn_pass True --irn_batch_size 32 --irn_crop_size 256 --irn_num_epoches 10 ^
    --irn_weights_name %IRN_WEIGHTS% ^
    --voc12_root %DATA_ROOT% ^
    --train_list %TRAIN_LIST% ^
    --ir_label_out_dir %IR_LABEL_OUT% ^
    --infer_list %DATA_ROOT% ^
    --irn_learning_rate 0.1

python run_sample.py --make_sem_seg_pass True --eval_sem_seg_pass True --sem_seg_bg_thres 0.4 ^
    --cam_out_dir %CAM_OUT% ^
    --sem_seg_out_dir %SEM_SEG_OUT% ^
    --irn_weights_name %IRN_WEIGHTS% ^
    --infer_list %TRAIN_LIST% ^
    --voc12_root %DATA_ROOT%

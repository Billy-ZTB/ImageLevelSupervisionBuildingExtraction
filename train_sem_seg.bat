python run_sample.py --train_semseg_pass True --pss_epochs 80 --pss_lr 0.007 --pss_wd 4e-5 ^
 --pss_results C:\ZTB\Results\potsdam\ACGC_0.4_0.3 ^
 --tag potsdam_ACGC_semseg_0.4_0.3 ^
    --voc12_root C:\ZTB\Dataset\VOC_potsdam ^
    --train_list C:\ZTB\Dataset\VOC_potsdam\train.txt ^
    --valid_list C:\ZTB\Dataset\VOC_potsdam\valid.txt ^
    --sem_seg_out_dir C:\ZTB\Code\ImageLevelSupervisionBuildingSegmentation\result\sem_seg_potsdam_0.4_0.3 ^
    --pss_batch_size 16 ^
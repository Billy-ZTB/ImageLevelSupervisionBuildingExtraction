python run_sample.py --train_pseudo True ^
--voc12_root C:\ZTB\Dataset\VOC_vaihingen ^
--train_list C:\ZTB\Dataset\VOC_vaihingen\train.txt ^
--valid_list C:\ZTB\Dataset\VOC_vaihingen\valid.txt ^
--pss_label_dir C:\ZTB\Code\ISSS\result\sem_seg_vaihingen_0.5_0.4 ^
--pss_crop_size 256 ^
--pss_batch_size 16 ^
--pss_num_epochs 100 ^
--tag pseudo_label_vaihingen
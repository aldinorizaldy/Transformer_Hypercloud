# FOR TRAINING

# train with LWIR (standardscaler dataset)
python train_transformer.py --model 'spct' --data_train 'data/train_lwir_degr_knn.npy' --epochs 100 --exp_name 'lwir_degr_spct_100epochs' --early_stop=False

# train with SWIR (standardscaler dataset)
python train_transformer.py --model 'spct' --data_train 'data/train_swir_degr_knn.npy' --epochs 100 --exp_name 'swir_degr_spct_100epochs' --early_stop=False

# train with VNIR (standardscaler dataset)
python train_transformer.py --model 'spct' --data_train 'data/train_vnir_degr_knn.npy' --epochs 100 --exp_name 'vnir_degr_spct_100epochs' --early_stop=False

# train with LWIR (standardscaler dataset)
python train_transformer.py --model 'naive_pct' --lr 0.00001 --data_train 'data/train_lwir_degr_knn.npy' --epochs 100 --exp_name 'lwir_degr_naive_pct_100epochs' --early_stop=False

# train with SWIR (standardscaler dataset)
python train_transformer.py --model 'naive_pct' --data_train 'data/train_swir_degr_knn.npy' --epochs 100 --exp_name 'swir_degr_naive_pct_100epochs' --early_stop=False

# train with VNIR (standardscaler dataset)
python train_transformer.py --model 'naive_pct' --lr 0.00001 --data_train 'data/train_vnir_degr_knn.npy' --epochs 100 --exp_name 'vnir_degr_naive_pct_100epochs' --early_stop=False

# train with LWIR (standardscaler dataset)
python train_transformer.py --model 'ascn_pct' --lr 0.00001 --data_train 'data/train_lwir_degr_knn.npy' --epochs 100 --exp_name 'lwir_degr_ascn_pct_100epochs' --early_stop=False

# train with SWIR (standardscaler dataset)
python train_transformer.py --model 'ascn_pct' --lr 0.00001 --data_train 'data/train_swir_degr_knn.npy' --epochs 100 --exp_name 'swir_degr_ascn_pct_100epochs' --early_stop=False

# train with VNIR (standardscaler dataset)
python train_transformer.py --model 'ascn_pct' --lr 0.00001 --data_train 'data/train_vnir_degr_knn.npy' --epochs 100 --exp_name 'vnir_degr_ascn_pct_100epochs' --early_stop=False

# train with LWIR (standardscaler dataset)
python train_transformer.py --model 'pt_pct' --lr 0.00001 --data_train 'data/train_lwir_degr_knn.npy' --epochs 100 --exp_name 'lwir_degr_pt_pct_100epochs' --early_stop=False

# train with SWIR (standardscaler dataset)
python train_transformer.py --model 'pt_pct' --lr 0.00001 --data_train 'data/train_swir_degr_knn.npy' --epochs 100 --exp_name 'swir_degr_pt_pct_100epochs' --early_stop=False

# train with VNIR (standardscaler dataset)
python train_transformer.py --model 'pt_pct' --lr 0.00001 --data_train 'data/train_vnir_degr_knn.npy' --epochs 100 --exp_name 'vnir_degr_pt_pct_100epochs' --early_stop=False

# FOR TESTING

# test with LWIR (standardscaler dataset)
python test_transformer.py --model 'spct' --data_test_folder 'data/test_lwir_degr_knn/' --model_path 'checkpoint/lwir_degr_spct_100epochs/model-best.pth' --exp_name 'XXX_test_lwir_degr_spct_100epochs'

# test with LWIR (standardscaler dataset)
python test_transformer.py --model 'spct' --data_test_folder 'data/test_lwir_degr_knn/' --model_path 'checkpoint/lwir_degr_spct_100epochs/model-best.pth' --exp_name 'test_lwir_degr_spct_100epochs'

# test with SWIR (standardscaler dataset)
python test_transformer.py --model 'spct' --data_test_folder 'data/test_swir_degr_knn/' --model_path 'checkpoint/swir_degr_spct_100epochs/model-best.pth' --exp_name 'test_swir_degr_spct_100epochs'

# test with VNIR (standardscaler dataset)
python test_transformer.py --model 'spct' --data_test_folder 'data/test_vnir_degr_knn/' --model_path 'checkpoint/vnir_degr_spct_100epochs/model-best.pth' --exp_name 'test_vnir_degr_spct_100epochs'

# test with LWIR (standardscaler dataset)
python test_transformer.py --model 'naive_pct' --data_test_folder 'data/test_lwir_degr_knn/' --model_path 'checkpoint/lwir_degr_naive_pct_100epochs/model-best.pth' --exp_name 'test_lwir_degr_naive_pct_100epochs'

# test with SWIR (standardscaler dataset)
python test_transformer.py --model 'naive_pct' --data_test_folder 'data/test_swir_degr_knn/' --model_path 'checkpoint/swir_degr_naive_pct_100epochs/model-best.pth' --exp_name 'test_swir_degr_naive_pct_100epochs'

# test with VNIR (standardscaler dataset)
python test_transformer.py --model 'naive_pct' --data_test_folder 'data/test_vnir_degr_knn/' --model_path 'checkpoint/vnir_degr_naive_pct_100epochs/model-best.pth' --exp_name 'test_vnir_degr_naive_pct_100epochs'

# test with LWIR (standardscaler dataset)
python test_transformer.py --model 'ascn_pct' --data_test_folder 'data/test_lwir_degr_knn/' --model_path 'checkpoint/lwir_degr_ascn_pct_100epochs/model-best.pth' --exp_name 'test_lwir_degr_ascn_pct_100epochs'

# test with SWIR (standardscaler dataset)
python test_transformer.py --model 'ascn_pct' --data_test_folder 'data/test_swir_degr_knn/' --model_path 'checkpoint/swir_degr_ascn_pct_100epochs/model-best.pth' --exp_name 'test_swir_degr_ascn_pct_100epochs'

# test with VNIR (standardscaler dataset)
python test_transformer.py --model 'ascn_pct' --data_test_folder 'data/test_vnir_degr_knn/' --model_path 'checkpoint/vnir_degr_ascn_pct_100epochs/model-best.pth' --exp_name 'test_vnir_degr_ascn_pct_100epochs'

# test with LWIR (standardscaler dataset)
python test_transformer.py --model 'pt_pct' --data_test_folder 'data/test_lwir_degr_knn/' --model_path 'checkpoint/lwir_degr_pt_pct_100epochs/model-best.pth' --exp_name 'test_lwir_degr_pt_pct_100epochs'

# test with SWIR (standardscaler dataset)
python test_transformer.py --model 'pt_pct' --data_test_folder 'data/test_swir_degr_knn/' --model_path 'checkpoint/swir_degr_pt_pct_100epochs/model-best.pth' --exp_name 'test_swir_degr_pt_pct_100epochs'

# test with VNIR (standardscaler datase
python test_transformer.py --model 'pt_pct' --data_test_folder 'data/test_vnir_degr_knn/' --model_path 'checkpoint/vnir_degr_pt_pct_100epochs/model-best.pth' --exp_name 'test_vnir_degr_pt_pct_100epochs'

# MERGE TESTING RESULTS

python final_prediction.py --pred_test_folder 'predict/test_lwir_degr_spct_100epochs'

python final_prediction.py --pred_test_folder 'predict/test_swir_degr_spct_100epochs'

python final_prediction.py --pred_test_folder 'predict/test_vnir_degr_spct_100epochs'

python final_prediction.py --pred_test_folder 'predict/test_lwir_degr_naive_pct_100epochs'

python final_prediction.py --pred_test_folder 'predict/test_swir_degr_naive_pct_100epochs'

python final_prediction.py --pred_test_folder 'predict/test_vnir_degr_naive_pct_100epochs'

python final_prediction.py --pred_test_folder 'predict/test_lwir_degr_ascn_pct_100epochs'

python final_prediction.py --pred_test_folder 'predict/test_swir_degr_ascn_pct_100epochs'

python final_prediction.py --pred_test_folder 'predict/test_vnir_degr_ascn_pct_100epochs'

python final_prediction.py --pred_test_folder 'predict/test_lwir_degr_pt_pct_100epochs'

python final_prediction.py --pred_test_folder 'predict/test_swir_degr_pt_pct_100epochs'

python final_prediction.py --pred_test_folder 'predict/test_vnir_degr_pt_pct_100epochs'
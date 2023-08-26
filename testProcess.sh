#!/usr/bin/env bash

python ~/NPS_s/scripts/preprocess_cafe.py `/bin/ls -v ~/cafe_gnn_s/MINI/ALE3DCAFE_PANDAS_aniso4_36/aniso4_36.*.dat` --dT=1 --data_slice=':,:,:78,:'
python -c "import numpy as np; import os; os.rename('out.npy','train4.npy'); d=np.load('train4.npy'); np.save('valid4.npy', d)" 

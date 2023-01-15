### 创建环境

    conda create -n slu python=3.6
    source activate slu
    pip install torch==1.7.1

### 模型说明
  包含 baseline 在内，目前该项目共包含 4 个网络模型来完成任务，它们分别是：
  + baseline(Bi-LSTM)
  + Bi-LSTM + CRF
  + BERT + CRF
  + BERT + Bi-LSTM + CRF

### 运行训练脚本  
在根目录下运行对应的命令
+ baseline(Bi-LSTM)

      python scripts/slu_baseline.py

+ Bi-LSTM + CRF

      python scripts/BiLSTM_CRF.py

+ BERT + CRF

      python scripts/BERT_CRF.py

+ BERT + Bi-LSTM + CRF

      python scripts/BERT_BiLSTM_CRF.py

### 运行测试脚本
+ baseline(Bi-LSTM)

      python scripts/slu_baseline_test_script.py
      
+ Bi-LSTM + CRF

      python scripts/BiLSTM_CRF_test_script.py

+ BERT + CRF

  **请注意：由于 bert 模型过大，所以没有上传 github，请在运行测试代码前先运行对应的训练代码**

      python scripts/BERT_CRF_test_script.py

+ BERT + Bi-LSTM + CRF

      python scripts/BERT_BiLSTM_CRF_test_script.py
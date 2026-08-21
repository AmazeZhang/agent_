wget -c -O ChartQA.zip "https://huggingface.co/datasets/ReFocus/ReFocus_Data/resolve/main/images/ChartQA.zip?download=true"
unzip ChartQA.zip
unzip train_chartqa_vcot.zip
python preprocess.py

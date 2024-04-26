@echo off
cd models

powershell -Command "Invoke-WebRequest -Uri 'https://drive.usercontent.google.com/download?id=179cuRZdJZEtEObovVf_KPhNFWnMF8pkN&confirm=xxx' -OutFile 'LENS-checkpoint.zip'"

tar -xf LENS-checkpoint.zip -C ./LENS
del LENS-checkpoint.zip
echo strict: False >> ./LENS/LENS/hparams.yaml

git clone https://github.com/yuh-zha/AlignScore.git
pip install AlignScore\.
python -m spacy download en_core_web_sm
powershell -Command "Invoke-WebRequest -Uri 'https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt' -OutFile './AlignScore/AlignScore-base.ckpt'"

cd ..
conda activate BiaPy_env
pip list --format=freeze > requirements.txt

sudo docker build -t danifranco/biapy:v1.0 .


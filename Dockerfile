FROM python:3.8.7-slim-buster

WORKDIR /
COPY . .

RUN pip install -r model-optimizer/requirements.txt

CMD python3 main_tf.py \
    --model_name $MODEL \
    --input_shape '(1, 64, 64, 16)' \
    --input_dir /opt/ml/processing/input_data \
    --output_dir /opt/ml/processing/processed_data

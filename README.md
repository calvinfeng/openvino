# OpenVino Fork

## Build Docker

```bash
docker build -t openvino-mo .
```

```bash
docker tag openvino-mo 812169894632.dkr.ecr.us-west-1.amazonaws.com/openvino-mo:dev
```

```bash
docker push 812169894632.dkr.ecr.us-west-1.amazonaws.com/openvino-mo:dev
```

```bash
docker run \
    -v /home/cfeng/py/openvino/input_data:/opt/ml/processing/input_data \
    -v /home/cfeng/py/openvino/processed_data:/opt/ml/processing/processed_data \
    -e "MODEL=final-64-64-16-unet" \
    openvino-mo
```
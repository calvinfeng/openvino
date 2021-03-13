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

## SageMaker

```python
import sagemaker
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput


s3_source = 's3://'
s3_destination = 's3://'


processor = Processor(image_uri='812169894632.dkr.ecr.us-west-1.amazonaws.com/openvino-mo:dev',
                     role=sagemaker.get_execution_role(),
                     instance_count=1,
                     instance_type="ml.m5.xlarge",
                     env={
                         'MODEL': '64-64-16-unet-final',
                     })

processor.run(inputs=[ProcessingInput(
                        source=s3_source,
                        destination='/opt/ml/processing/input_data')
                     ],
              outputs=[ProcessingOutput(
                        output_name="openvino-ir",
                        source='/opt/ml/processing/processed_data',
                        destination=s3_destination)
                      ])
```
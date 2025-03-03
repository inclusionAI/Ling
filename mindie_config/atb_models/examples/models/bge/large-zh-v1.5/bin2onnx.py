# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from optimum.onnxruntime import ORTModelForFeatureExtraction
from atb_llm.models.base.model_utils import safe_from_pretrained

parser = argparse.ArgumentParser(description="Export a model from transformers to ONNX format.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint to convert.")

args = parser.parse_args()

model_checkpoint = args.model_path

ort_model = safe_from_pretrained(ORTModelForFeatureExtraction, model_checkpoint, export=True, from_transformers=True)

# Save the ONNX model
ort_model.save_pretrained(model_checkpoint)
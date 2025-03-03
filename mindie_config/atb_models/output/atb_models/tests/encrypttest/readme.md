# 免责声明！！！

## 当前仅仅提供权重加解密的一个范例，不对安全性进行负责，用户需根据具体场景使用自己的加密脚本进行替换。

# 模型加解密使用指导

## 加解密脚本介绍

### encrypt.py

将加密相关的信息放在自定义加密类中；加密算法在类中的encrypt方法中实现

```python
class EncryptTools(Encrypt):
    def __init__(self) -> None:
        super().__init__()

        # generate key
        self.key = your key info

    def encrypt(self, tensor: torch.Tensor):
        """
        输入是原始tensor，输出是加密tensor。
        保证加密前后，encrypted_tensor和tensor的shape一致。
        """
	return encrypted_tensor
```

本仓库给出脚本中的key生成只是一种使用范例，加密算法使用的AES-256，CTR模式。

若使用本仓库的AES加密算法，请先安装 `cryptography` 。

```bash
pip install cryptography
```

### decrypt.py

将解密相关的信息放在自定义解密类中；解密算法在类中的decrypt方法中实现

```python
class DecryptTools(Decrypt):
    def __init__(self) -> None:
        super().__init__()

        # gain key
        self.key = your key info

    def decrypt(self, encrypted_tensor: torch.Tensor):
        """
        输入是加密tensor，输出是解密tensor。 
        保证解密前后，encrypted_tensor和decrypted_tensor的shape一致。
        """
	return decrypted_tensor
```

本仓库给出脚本中的key只是一种使用范例，解密算法使用的AES-256，CTR模式。

## 加密阶段

使用脚本 `encrypt_weights.py` ，输入需要加密的模型权重路径和加密后的权重路径。

在用户的真实场景中，加密阶段一般在用户设备进行，所生成的密钥等信息也由用户决定设置在具体的存储位置，密钥等信息是存储在用户自己的设备上的。

**注意事项：**

1. 选择安全的加密算法，例如：AES-256, CTR模式加密算法；
2. 密钥不要用明文存储；
3. 对权重和密钥文件存储路径进行合法性和安全性校验。

### 模型加密运行脚本示例

```bash
cd ${llm_path}
python tests/encrypttest/encrypt_weights.py --model_weights_path /your/model/path --encrypted_model_weights_path /your/encrypt_model/path --key_path /your/key/path
```

若想使用自定义加密算法，修改加密脚本 `encrypt.py` 中的加密函数。

## 解密推理阶段

1. 需要将 解密脚本 `decrypt.py`  复制到 `atb_llm/utils/` 路径下。
2. 使用脚本 `decrypt_code_padding.sh` 给 `${llm_path}atb_llm/utils/` 路径下的 `weights.py` 文件增加解密代码。使用方法如下：

   ```bash
   bash decrypt_code_padding.sh ${llm_path}/atb_llm/utils/weights.py
   ```

完成上述操作之后，即可实现使用加密后的模型权重进行推理。同时解密脚本中含有针对权重文件的解密方法接口，用户可以传入加密权重文件路径将加密权重进行解密。当然，针对文件的具体解密算法，用户需重新实现DecryptTools类中的 `decrypt`方法。

#### 纯推理 运行脚本示例：

```bash
# 使用多卡运行Paged Attention，设置模型权重路径，设置输出长度为2048个token
cd ${llm_path}
torchrun --nproc_per_node 2 --master_port 20038 -m examples.run_pa --model_path ${encrypt_weight_path} --max_output_length 2048 --kw_args '{"encrypt_enable": 1, "key_path": "/your/key/path"}'
```

其中，

- encrypt_enable 参数控制是否可以加载加密的模型权重，默认为0。
- key_path  密钥路径。

其余参数的解释请参考 `run_pa` 的[介绍文档](https://gitee.com/ascend/MindIE-LLM/blob/master/examples/atb_models/examples/README.md)。

#### 使用下游问答数据集的精度测试例子。

使用下游数据集进行测试时，先下载好数据集。

```bash
##单机场景
cd ${llm_path}/tests/modeltest
bash run.sh pa_[data_type] [dataset] ([shots]) [batch_size] [model_name] ([is_chat_model]) (lora [lora_data_path]) [weight_dir] ([encrypt_info]) [chip_num] ([max_position_embedding/max_sequence_length])

## 示例
bash run.sh pa_fp16 full_TruthfulQA 4 llama /your/model/path '{"encrypt_enable": 1, "key_path": "/your/key/path"}' 8

```

其中，encrypt_enable 参数控制是否可以加载加密的模型权重， key_path 表示加密权重的密钥路径。

其余参数的解释请参考modeltest下的[介绍文档](https://gitee.com/ascend/MindIE-LLM/blob/master/examples/atb_models/tests/modeltest/README.md) 中的  精度测试（下游数据集) 章节。

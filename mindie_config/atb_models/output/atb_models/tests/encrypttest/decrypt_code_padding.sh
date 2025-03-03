#!/bin/bash
file_path=$1

cp $file_path $file_path'_bak'

# 添加包
sed -i '/import os/a\import importlib' $file_path

# 读取解密接口
sed -i "/routing = self.load_routing(process_group)/i\\
        self.encrypt_enable = kwargs.get('encrypt_enable', 0)\n\
        if self.encrypt_enable:\n\
            # gain decrypt func\n\
            key_path = kwargs.get('key_path', None)\n\
            key_path = file_utils.standardize_path(key_path, check_link=False)\n\
            file_utils.check_path_permission(key_path)\n\
            decrypt_script = importlib.import_module('atb_llm.utils.decrypt')\n\
            decrypt_cls = getattr(decrypt_script, 'DecryptTools')\n\
            self.decrypt_ins = decrypt_cls(**kwargs)\n\
\n\
        self.sf_metadata = {}" $file_path

sed -i "/for k in f.keys():/i\                self.sf_metadata.update(f.metadata())" $file_path

sed -i 's/        tensor = f.get_tensor(tensor_name)/        if self.encrypt_enable:\
            # 使用tensor name 获取加密tensor\
            tensor = f.get_tensor(tensor_name)\
            # 获取解密 tensor  \
            tensor = self.decrypt_ins.decrypt(tensor)\
            # sf_metadata 中找到加密前的数据类型并做转换\
            if tensor_name in self.sf_metadata:\
                module_name, attribute_name = self.sf_metadata[tensor_name].split(".")\
                module = importlib.import_module(module_name)\
                dtype_ = getattr(module, attribute_name)\
            else:\
                raise AssertionError(f"{tensor_name} does not exist in metadata")\
            tensor = tensor.to(dtype_)\
        else:\
            tensor = f.get_tensor(tensor_name)\
        del self._handles[filename]/g' $file_path


sed -i '/stop = slice_.get_shape()\[dim\]/a\
        if self.encrypt_enable:\
            slice_ = self.get_tensor(tensor_name)' $file_path

sed -i '/if "c_attn.bias" in tensor_name:/i\
        if self.encrypt_enable:\
            slice_ = self.get_tensor(tensor_name)\
' $file_path

sed -i '/head_num = size \/\/ gqa_size/a\
        if self.encrypt_enable:\
            slice_ = self.get_tensor(tensor_name)\
' $file_path

sed -i ":a;N;s/def get_tensor_col_packed_qkv_mha(self, tensor_name: str, head_size: int = None, dim=0):\n        slice_ = self._get_slice(tensor_name)/temp\/to\/replace/g;ta" $file_path
sed -i 's/temp\/to\/replace/def get_tensor_col_packed_qkv_mha(self, tensor_name: str, head_size: int = None, dim=0):\
        slice_ = self._get_slice(tensor_name)\
        slice_shape = slice_.get_shape()\
        if self.encrypt_enable:\
            slice_ = self.get_tensor(tensor_name)/g' $file_path

sed -i 's/total_size = slice_.get_shape()\[-1\]/total_size = slice_shape[-1]/g' $file_path

sed -i 's/if len(slice_.get_shape()) <= 1:/if len(slice_shape) <= 1:/g' $file_path

sed -i 's/q_zero = torch.zeros(size=(rank_heads \* head_size, slice_.get_shape()\[1\]))/q_zero = torch.zeros(size=(rank_heads * head_size, slice_shape[1]))/g' $file_path

sed -i 's/k_zero = torch.zeros(size=(rank_heads \* head_size, slice_.get_shape()\[1\]))/k_zero = torch.zeros(size=(rank_heads * head_size, slice_shape[1]))/g' $file_path

sed -i 's/v_zero = torch.zeros(size=(rank_heads \* head_size, slice_.get_shape()\[1\]))/v_zero = torch.zeros(size=(rank_heads * head_size, slice_shape[1]))/g' $file_path

sed -i 's/total_size = slice_.get_shape()\[0\]/total_size = slice_shape[0]/g' $file_path

sed -i ":a;N;s/def get_tensor_col_packed_kv_mha(self, tensor_name: str, hiden_size, head_size: int = None):\n        slice_ = self._get_slice(tensor_name)/temp\/to\/replace/g;ta" $file_path
sed -i 's/temp\/to\/replace/def get_tensor_col_packed_kv_mha(self, tensor_name: str, hiden_size, head_size: int = None):\
        slice_ = self._get_slice(tensor_name)\
        slice_shape = slice_.get_shape()\
        if self.encrypt_enable:\
            slice_ = self.get_tensor(tensor_name)/g' $file_path


sed -i 's/if len(slice_.get_shape()) == 1:/if len(slice_shape) == 1:/g' $file_path

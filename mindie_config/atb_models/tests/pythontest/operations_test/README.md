# MindIE LLM Operation Test

## 介绍
Operation Test是基于python组图架构的算子测试

## 代码框架
operation_test 继承了python中unittest，并提供了基于python组图的测试接口(run_compare),后续新增的op测试可以调用此接口测试。

其他文件夹下包含了一些算子的测试用例

## 使用说明
算子测试使用对应目录下的python文件，会对比基于torch构建的golden和算子进行比较 返回assertTrue

## 未来更新帮助
1.算子测试需要自行构建随机输入
2.自行构建golden函数
3.run_compare接口说明：op_type:需要测试算子的名称
                      op_param:算子的参数（字典形式）
                      op_name:个人理解没什么用，但是atb._BaseOperation中一定要传参
                      in_tensor：算子的传参（字典形式）传参的输入的键命名只能'in0'、'in1' 以此类推，输出字典同理（'out0'）
                      out_tensor：算子输出（字典形式）
4.算子输入、输出和实现原理请参考：https://www.hiascend.com/document/detail/zh/mindie/1.0.RC1/mindiert/rtdev/ascendtb_01_0044.html


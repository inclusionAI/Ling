# HumanEval_X数据集配置依赖环境 README


# 环境安装
在测试HumanEval_X时，可参照scripts/install_humaneval_x_dependency.sh配置依赖环境
1.首先需要设置代理，根据服务器网段设置服务器代理
2.source scripts/install_humaneval_x_dependency.sh
3.导入环境变量，否则可能跑不出结果，影响平均精度 export PATH=/usr/local/go/bin:$PATH
4.检查语言是否安装成功
    检查go语言：go version
    检查js：js --version
    检查java：java --version
    检查npm：npm -v
若语言出现对应的版本说明下载安装成功，即可进行精度测试
若没有出现对应版本则需要重新下载或者单独安装相关的语言


# 其他问题定位：
1.如出现某个语言精度结果为0.0则应检查对应的语言版本是否下载完成，并查看是否导入环境变量
2.go语言和JAVA的执行时间比较久，如果发现这两个语言有波动，则可以检查modeltest/task/humaneval_x.py文件中check_correctness方法中的timeout设置是否太小，导致用例超时，可以设置为50。
3.transformers版本是否符合，可以根据config文件查询对应模型的transformers版本

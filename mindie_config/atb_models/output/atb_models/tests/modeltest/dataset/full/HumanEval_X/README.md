# HumanEval_X数据集配置依赖环境 README


## `HumanEval_X`环境安装
在测试HumanEval_X时，需要配置语言环境，包括GO、JS、JAVA这三种语言,安装步骤如下：


1.首先需要设置代理，根据服务器网段设置服务器代理

2.下载语言包：
    
    NPU环境上的语言包为：
    go    https://go.dev/dl/go1.18.4.linux-arm64.tar.gz
    node  https://nodejs.org/download/release/v16.14.0/node-v16.14.0-linux-arm64.tar.gz
    java  https://download.oracle.com/java/18/archive/jdk-18.0.2.1_linux-aarch64_bin.tar.gz

    GPU环境上的语言包为：
    go    https://go.dev/dl/go1.18.4.linux-amd64.tar.gz
    node  https://nodejs.org/download/release/v16.14.0/node-v16.14.0-linux-x64.tar.gz
    java  https://download.oracle.com/java/18/archive/jdk-18.0.2.1_linux-x64_bin.tar.gz

3.执行解压安装：
    
    NPU安装步骤：

        # 安装npm:可能代理网络不好，导致npm下载失败，可以更换代理或者换个时间段在进行安装
        apt-get update
        apt-get install -y npm

        # 安装GO语言：
        tar -zxf go1.18.4.linux-arm64.tar.gz -C /usr/local
        export PATH=/bin:/usr/local/go/bin:$PATH

        # 安装node:
        mkdir -p /usr/local/lib/nodejs
        tar -zxf node-v16.14.0-linux-arm64.tar.gz -C /usr/local/lib/nodejs
        mv /usr/local/lib/nodejs/node-v16.14.0-linux-arm64 /usr/local/lib/nodejs/node

        # 安装js:
        npm config set strict-ssl false
        npm install -g js-md5@0.7.3
        export PATH=/usr/local/lib/nodejs/node/bin:$PATH
        export NODE_PATH=/usr/local/lib/node_modules

        # 安装JAVA:
        mkdir /usr/java
        tar -zxf jdk-18.0.2.1_linux-aarch64_bin.tar.gz -C /usr/java
        JAVA_HOME=/usr/java/jdk-18.0.2.1
        update-alternatives --install /usr/bin/java java $JAVA_HOME/bin/java 20000
        update-alternatives --install /usr/bin/javac javac $JAVA_HOME/bin/javac 20000

    GPU：与NPU类似，只需要更换上述脚本中对应的语言安装包名即可

4.检查语言是否安装成功

    go version  #检查go语言
    js --version  #检查js
    java --version  #检查java
    npm -v  #检查npm

若语言出现对应的版本说明下载安装成功，即可进行精度测试
若没有出现对应版本则需要重新下载或者单独安装相关的语言


## `HumanEval_X`环境配置补充说明

1.如出现某个语言精度结果为0.0则应检查对应的语言版本是否下载完成，并查看是否导入环境变量

2.go语言和JAVA的执行时间比较久，如果发现这两个语言有波动，则可以检查model_test.py中的timeout设置是否太小。

3.每次进入环境都需要进行导入GO语言环境，检查语言版本



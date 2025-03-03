# 💻 `HumanEval_X`数据集任务的环境配置

在测试`HumanEval_X`时，需要额外安装和配置多语言环境。请参考以下步骤完成配置：

## 安装`curl`和`npm`

- 首先确保`curl`和`npm`已安装，可以通过以下命令进行安装：

```bash
apt-get install -y curl npm
```

## `Go`语言安装

- 首先需要下载并安装`Go`语言（使用 [v1.18.4 版本](https://go.dev/dl/go1.18.4.linux-arm64.tar.gz)）。
- 安装完成后，将`Go`二进制路径添加到系统的`PATH`环境变量中，以确保可以全局调用`Go`命令。

```bash
# 请将 INSTALL_PATH 替换为系统中常用的安装路径（例如 /opt 或 /usr/local）
export PATH=${INSTALL_PATH}/go/bin:$PATH
```

## `JavaScript`环境安装

- 下载并安装`Node.js`（使用 [v16.14.0 版本](https://nodejs.org/download/release/v16.14.0/node-v16.14.0-linux-arm64.tar.gz)）。
- 安装成功后，需要安装`js-md5`包，并配置`Node.js`的执行路径到系统的`PATH`中。

```bash
# 请将 INSTALL_PATH 替换为系统中常用的安装路径（例如 /opt 或 /usr/local）
npm install -g js-md5@0.7.3 # 使用 npm 安装 js-md5 包
export PATH=${INSTALL_PATH}/nodejs/node/bin:$PATH # 将 Node.js 的路径添加到 PATH 环境变量
export NODE_PATH=${INSTALL_PATH}/node_modules # 设置 NODE_PATH 变量
```

## `Java`环境安装

- 下载并安装`JDK`（使用 [JDK 18.0.2.1 版本](https://download.oracle.com/java/18/archive/jdk-18.0.2.1_linux-aarch64_bin.tar.gz)）。
- 安装后设置`JAVA_HOME`环境变量，并确保通过`update-alternatives`命令将`Java`和`Javac`绑定到正确的可执行路径。

```bash
# 请将 INSTALL_PATH 替换为系统中常用的安装路径（例如 /opt 或 /usr/local）
export JAVA_HOME=${INSTALL_PATH}/jdk-18.0.2.1
echo "export JAVA_HOME=${INSTALL_PATH}/jdk-18.0.2.1" >> ~/.profile
# 使用 update-alternatives 命令配置 Java 和 Javac
update-alternatives --install /usr/bin/java java $JAVA_HOME/bin/java 20000
update-alternatives --install /usr/bin/javac javac $JAVA_HOME/bin/javac 20000
```

## 环境变量导入

确保`Go`、`JavaScript`、`Java`的可执行文件路径已正确配置到`PATH`环境变量中：

```bash
# 请将 INSTALL_PATH 替换为系统中常用的安装路径（例如 /opt 或 /usr/local）
export PATH=${INSTALL_PATH}/go/bin:$PATH # Go
which go # 确保Go PATH配置正确
export PATH=${INSTALL_PATH}/lib/nodejs/node/bin:$PATH # JavaScript
which nodejs # 确保Node.js PATH配置正确
export JAVA_HOME=${INSTALL_PATH}/java/jdk-18.0.2.1 # Java，并在 .profile 中设置。
which java # 确保JAVA PATH配置正确
```

## 检查安装状态

- 验证`Go`、`JavaScript`、`Java`环境是否安装成功：
    - 运行 `go version` 检查`Go`语言版本。
    - 运行 `node --version` 检查`Node.js`版本。
    - 运行 `java --version` 检查`Java`版本。
    - 运行 `npm -v` 检查`npm`版本。

## 代理设置

- 在某些服务器环境中，可能需要设置代理，以确保可以下载和安装所需的依赖包。请根据服务器的网络设置适当配置代理。

## `HumanEval_X`环境配置补充说明

1. 运行`HumanEval_X`时的语言支持

    - `HumanEval_X`任务涉及多语言的代码评测，包括`Go`、`Java`、`JavaScript`等。因此，必须确保所有语言环境均已正确安装，并且可通过系统的`PATH`环境变量调用。

2. 超时设置

    - 如果执行`Go`和`Java`任务时耗时较长，可能需要检查`HumanEval_X`的`check_correctness`方法中的超时设置。可以适当增大`timeout`值，建议将其设置为`50`秒，避免任务超时导致结果为`0`。

3. 版本兼容性

    - 请确保`Transformers`库的版本与模型的要求一致，可以通过配置文件中查询模型对应的版本。如果`Transformers`版本不匹配，可能会导致测试无法正常运行。

4. 其他问题定位

    - 如果测试某种语言时出现精度为`0.0`，建议检查语言环境是否安装正确，是否导入了相关的环境变量。如果安装有问题，可以参考上面的步骤重新配置相关环境。
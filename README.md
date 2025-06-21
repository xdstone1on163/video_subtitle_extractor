# 视频字幕工具

这是一个综合性的视频字幕工具，集成了视频字幕获取和字幕截图文字识别功能，利用AWS云服务（Bedrock、Transcribe、Translate、S3）实现高效的视频字幕处理流程。
这是视频功能演示：


https://github.com/user-attachments/assets/96d97bba-6862-4935-8ad4-cf7d39b5af53



## 主要功能

### 1. 视频字幕获取
- 浏览S3存储桶中的视频文件
- 使用AWS Transcribe服务自动转录视频中的语音
- 支持多种语言的语音识别（法语、德语、日语等）
- 生成SRT和VTT格式的字幕文件
- 自动将字幕翻译成中文
- 提供字幕文件下载和预览功能

### 2. 字幕截图文字识别
- 上传本地视频并提取指定区域的帧
- 支持自定义截取区域和截图频率
- 将提取的帧上传到S3存储
- 浏览S3存储桶中的图片
- 使用AWS Bedrock的大语言模型识别图片中的文字
- 支持多种语言的字幕识别（SA、JP、KR、FR、IT、DE、UA、TR）
- 支持多种AWS Bedrock模型（Claude 3系列和Nova系列）
- 自动检查原文语法和拼写错误
- 将识别的字幕翻译成中文

### 3. S3路径设置
- 统一配置视频存储路径和上传目录路径
- 加载设置到各个功能模块，实现无缝集成

## Amazon Linux 2023环境部署步骤

### 前提条件
- Amazon Linux 2023实例（EC2或本地环境）
- AWS账户，并已启用Bedrock、Transcribe和S3服务权限
- 有权访问的S3存储桶，用于存储视频和字幕截图

### 步骤1: 准备环境

```bash
# 更新系统包
sudo dnf update -y

# 安装Python和pip（Amazon Linux 2023已预装Python 3.9）
sudo dnf install python3-pip python3-devel -y

# 安装视频处理所需的系统依赖
sudo dnf install ffmpeg opencv-devel mesa-libGL -y
```

### 步骤2: 创建并激活虚拟环境

```bash
# 创建项目目录
mkdir -p ~/video_subtitle_tool
cd ~/video_subtitle_tool

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

### 步骤3: 获取代码并安装依赖

```bash
# 克隆代码仓库（如果使用Git）
# git clone <repository-url> .

# 或者手动上传代码到服务器

# 安装依赖
pip install -r requirements.txt
```

### 步骤4: 配置AWS凭证

确保已经配置好AWS凭证，可以通过以下方式之一：

**方式1: 使用AWS CLI配置**

```bash
# 安装AWS CLI（如果尚未安装）
pip install awscli

# 配置凭证
aws configure
```

然后输入您的AWS访问密钥ID、秘密访问密钥、默认区域（建议使用`us-west-2`，因为代码中使用了该区域）和输出格式。

**方式2: 设置环境变量**

```bash
export AWS_ACCESS_KEY_ID=你的访问密钥ID
export AWS_SECRET_ACCESS_KEY=你的秘密访问密钥
export AWS_DEFAULT_REGION=us-west-2
```

要使这些设置持久化，可以将它们添加到`~/.bashrc`或`~/.profile`文件中。

### 步骤5: 启动应用

```bash
# 确保在虚拟环境中
source venv/bin/activate

# 启动应用
python app.py
```

默认情况下，应用将在本地运行，通常可以通过浏览器访问 http://127.0.0.1:7860 打开界面。

### 步骤6: 设置为后台服务（可选）

如果需要将应用作为后台服务运行，可以使用`systemd`或`screen`：

**使用screen**:

```bash
# 安装screen
sudo dnf install screen -y

# 创建新的screen会话
screen -S subtitle_tool

# 在screen会话中启动应用
source venv/bin/activate
python app.py

# 按Ctrl+A然后按D，将会话分离到后台
```

**使用systemd**:

创建服务文件：

```bash
sudo nano /etc/systemd/system/subtitle-tool.service
```

添加以下内容：

```
[Unit]
Description=Video Subtitle Tool
After=network.target

[Service]
User=<your-username>
WorkingDirectory=/home/<your-username>/video_subtitle_tool
ExecStart=/home/<your-username>/video_subtitle_tool/venv/bin/python app.py
Restart=on-failure
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

启用并启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable subtitle-tool
sudo systemctl start subtitle-tool
```

### 步骤7: 设置公网访问（可选）

如果需要从公网访问应用，可以直接使用Gradio的内置功能：

**方法1: 使用Gradio临时共享链接**

修改`app.py`文件中的`app.launch()`行：

```python
app.launch(share=True)
```

这将生成一个临时的公共URL，可以从任何地方访问，但URL会在每次重启应用后改变。

**方法2: 让Gradio监听所有网络接口**

修改`app.py`文件中的`app.launch()`行：

```python
app.launch(server_name="0.0.0.0")
```

这将使应用监听所有网络接口，可以通过服务器的IP地址或域名访问（如果已配置DNS）。

**注意事项**:
- 确保EC2安全组或防火墙已开放7860端口（Gradio默认端口）
- 如果使用自定义端口，可以添加`server_port`参数：`app.launch(server_name="0.0.0.0", server_port=8080)`
- 对于生产环境，建议配置SSL证书以启用HTTPS

## 使用指南

### 视频字幕获取
1. 在左侧导航菜单中选择"视频字幕获取"
2. 在"S3视频存储路径"文本框中输入您的S3存储桶路径
3. 点击"浏览S3视频"按钮加载视频列表
4. 在视频列表中选择需要处理的视频
5. 选择转录语言
6. 点击"Transcribe视频语音"按钮开始转录
7. 使用"检查任务状态"按钮查看转录进度
8. 转录完成后，可以下载SRT和VTT格式的字幕文件，并查看转录和翻译结果

### 字幕截图文字识别
1. 在左侧导航菜单中选择"字幕截图文字识别"
2. 上传本地视频或使用S3中的视频
3. 设置截取区域的坐标和尺寸
4. 设置截图频率
5. 点击"开始截取"按钮提取视频帧
6. 查看提取的帧，可以删除不需要的帧
7. 在"S3上传目录"中输入上传路径，点击"上传到S3"按钮
8. 在"待处理截图的S3存储路径"中输入路径，点击"浏览S3图片"按钮
9. 在图库中选择需要识别的字幕图片
10. 选择合适的Bedrock模型和字幕语言
11. 点击"图片处理"按钮开始OCR处理
12. 识别结果将显示在"识别文字结果"文本框中

## 语言代码对照表

| 代码 | 语言 |
|------|------|
| SA | 梵文/阿拉伯语 |
| JP | 日语 |
| KR | 韩语 |
| FR | 法语 |
| IT | 意大利语 |
| DE | 德语 |
| UA | 乌克兰语 |
| TR | 土耳其语 |

## 故障排除

- 如果遇到AWS凭证错误，请检查凭证设置是否正确，以及是否有足够的权限
- 如果无法获取S3文件列表，请检查S3路径格式和访问权限
- 如果OCR结果不准确，尝试调整为不同的模型，或确保图片质量清晰
- 如果视频转录失败，检查视频格式是否受支持，以及AWS Transcribe服务是否在您的账户中启用

## 注意事项

- 使用AWS服务（Bedrock、Transcribe、S3等）会产生相应费用，请注意控制使用量
- 建议使用低延迟、高质量的图片以获得最佳OCR效果
- 某些语言可能在特定模型上效果更好，可以尝试不同组合
- 大型视频文件的转录可能需要较长时间，请耐心等待

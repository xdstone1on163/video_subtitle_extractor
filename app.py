import gradio as gr
import boto3
from PIL import Image
import io
import base64
import json
import os
import cv2
import numpy as np
import tempfile
import uuid
import time
from pathlib import Path
from datetime import datetime

def get_model_id(model_name):
    """根据界面选择的模型名称返回Bedrock模型ID"""
    model_mapping = {
        "Claude 3 Opus": "us.anthropic.claude-3-opus-20240229-v1:0",
        "Claude 3 Sonnet": "us.anthropic.claude-3-sonnet-20240229-v1:0",
        "Claude 3.5 Haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "Claude 3.5 Sonnet v2": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "Claude 3.5 Sonnet v1": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "Claude 3.7 Sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "Nova Lite": "us.amazon.nova-lite-v1:0",
        "Nova Pro": "us.amazon.nova-pro-v1:0"
    }
    return model_mapping.get(model_name)

def get_language_name(lang_code):
    """将语言代码转换为完整名称"""
    language_map = {
        "SA": "梵文/阿拉伯语",
        "JP": "日语",
        "KR": "韩语",
        "FR": "法语",
        "IT": "意大利语",
        "DE": "德语",
        "UA": "乌克兰语",
        "TR": "土耳其语"
    }
    return language_map.get(lang_code, "未知语言")

def list_s3_videos(s3_path):
    """列出S3路径下的所有视频文件并返回用于展示的格式"""
    try:
        if not s3_path:
            return [], []
            
        # 处理s3://前缀
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]  # 移除's3://'前缀
            
        # 解析bucket和prefix
        parts = s3_path.strip('/').split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        
        # 创建S3客户端
        s3_client = boto3.client('s3')
        
        # 列出对象
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
        
        # 筛选视频文件并生成预览URL
        video_list = []
        metadata_list = []
        
        for item in response.get('Contents', []):
            if item['Key'].lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.wmv')):
                # 生成预签名URL，有效期更长以支持视频播放
                url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket, 'Key': item['Key']},
                    ExpiresIn=86400  # 24小时
                )
                video_list.append(url)
                metadata_list.append({
                    "key": item['Key'],
                    "name": item['Key'].split('/')[-1],
                    "size": item['Size']
                })
        
        return [video_list, metadata_list]  # 返回视频URL列表和元数据列表
    
    except Exception as e:
        print(f"S3视频列表错误: {str(e)}")
        return [[f"错误: {str(e)}"], [{"key": "error"}]]

def list_s3_images(s3_path):
    """列出S3路径下的所有图片并返回用于Gallery展示的格式"""
    try:
        if not s3_path:
            return [], []
            
        # 处理s3://前缀
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]  # 移除's3://'前缀
            
        # 解析bucket和prefix
        parts = s3_path.strip('/').split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        
        # 创建S3客户端
        s3_client = boto3.client('s3')
        
        # 列出对象
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
        
        # 筛选图片文件并生成预览URL
        image_list = []
        metadata_list = []
        
        for item in response.get('Contents', []):
            if item['Key'].lower().endswith(('.png', '.jpg', '.jpeg')):
                # 生成预签名URL
                url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket, 'Key': item['Key']},
                    ExpiresIn=3600
                )
                image_list.append(url)
                metadata_list.append({"key": item['Key']})
        
        return [image_list, metadata_list]  # 返回图片URL列表和元数据列表
    
    except Exception as e:
        print(f"S3列表错误: {str(e)}")
        return [[f"错误: {str(e)}"], [{"key": "error"}]]

def select_image(evt, s3_path):
    """处理图片选择事件，加载选中的图片"""
    try:
        if not evt or not s3_path:
            return None
            
        # 从事件中获取选中的图片Key
        image_key = evt
        
        # 处理s3://前缀
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]  # 移除's3://'前缀
            
        # 解析bucket和key
        parts = s3_path.strip('/').split('/', 1)
        bucket = parts[0]
        
        # 创建S3客户端并下载图片
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=image_key)
        
        # 转换为PIL图像
        image_data = response['Body'].read()
        image = Image.open(io.BytesIO(image_data))
        
        return image
        
    except Exception as e:
        print(f"选择图片错误: {str(e)}")
        return None

def extract_text(image, model_name, language, system_prompt, user_prompt):
    """使用Bedrock提取图像中的文字"""
    try:
        if image is None:
            return "请先选择一个图片"
            
        # 获取模型ID
        model_id = get_model_id(model_name)
        
        # 创建Bedrock客户端
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-west-2'  # 使用us-west-2区域进行跨区域调用
        )
        
        # 准备提示词
        language_name = get_language_name(language)
        # 将语言信息添加到用户提示中
        full_user_prompt = f"{user_prompt}\n语言: {language_name}"
        
        # 回到之前使用的invoke_model方法，因为它对Claude是有效的
        if "Claude" in model_name:
            # 将图像转换为base64编码
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Claude模型使用invoke_model API，这是已知可用的方法
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": full_user_prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            }
            
            print(f"使用Claude模型: {model_id}")
            
            # 发送请求
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            # 解析响应
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
            
        else:
            # Nova模型使用invoke_model API直接处理图像，但使用不同的请求结构
            print(f"使用Nova模型: {model_id}")
            
            try:
                # 将图像转换为PNG格式
                buffered = io.BytesIO()
                # 如果是RGBA格式(带透明度的PNG)，则转换为RGB
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                image.save(buffered, format="PNG")
                image_bytes = buffered.getvalue()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                
                # 根据用户提供的示例构建请求
                # 系统消息
                system_list = [
                    {
                        "text": system_prompt
                    }
                ]
                
                # 用户消息，先放图像再放文本
                message_list = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "png",
                                    "source": {
                                        "bytes": image_base64
                                    }
                                }
                            },
                            {
                                "text": full_user_prompt
                            }
                        ]
                    }
                ]
                
                # 推理配置
                inf_params = {
                    "maxTokens": 1000,
                    "temperature": 0.1,
                    "topP": 0.9,
                    "topK": 50
                }
                
                # 构建完整请求
                request_body = {
                    "schemaVersion": "messages-v1",
                    "messages": message_list,
                    "system": system_list,
                    "inferenceConfig": inf_params
                }
                
                # 打印调试信息，但不包含图像数据
                debug_request = {
                    "schemaVersion": "messages-v1",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": {
                                        "format": "png",
                                        "source": {"bytes": "[图像数据已省略]"}
                                    }
                                },
                                {
                                    "text": full_user_prompt
                                }
                            ]
                        }
                    ],
                    "system": system_list,
                    "inferenceConfig": inf_params
                }
                print(f"Nova请求结构(不含图像数据): {json.dumps(debug_request)}")
                
                # 使用invoke_model API
                response = bedrock_runtime.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body)
                )
                
                # 解析响应
                response_body = json.loads(response['body'].read())
                print(f"Nova响应结构: {json.dumps(response_body)}")
                
                # 从响应中提取文本
                if 'output' in response_body and 'message' in response_body['output'] and 'content' in response_body['output']['message']:
                    content = response_body['output']['message']['content']
                    if isinstance(content, list) and len(content) > 0 and 'text' in content[0]:
                        return content[0]['text']
                
                # 如果无法找到预期的结构，返回完整响应
                return f"完整Nova响应: {json.dumps(response_body)}"
                    
            except Exception as nova_error:
                error_message = f"Nova API错误: {str(nova_error)}"
                print(f"Nova调用错误: {str(nova_error)}")
                return error_message
        
    except Exception as e:
        return f"错误: {str(e)}"

def create_subtitle_recognition_ui():
    """创建字幕截图文字识别界面"""
    with gr.Column() as subtitle_ui:
        # 创建状态变量存储图片元数据
        image_keys = gr.State([])
        area_selection = gr.State({"x": 120, "y": 1300, "width": 850, "height": 220})
        
        # 全局变量存储提取的帧路径
        global extracted_frame_paths
        
        # 视频上传与播放区域
        with gr.Row():
            with gr.Column():
                # 文件上传和播放组件
                upload_video = gr.Video(
                    label="上传或播放视频（可直接拖拽上传本地视频）",
                    interactive=True,
                    height=400
                )
                
                # 添加视频分辨率显示
                video_dimensions = gr.HTML(
                    """
                    <div style="text-align: center; margin-top: 10px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
                        <h3 style="margin: 0;">视频分辨率</h3>
                        <p id="video-dimensions-text" style="font-size: 16px; margin: 5px 0;">请上传视频以查看分辨率信息</p>
                    </div>
                    """
                )

        # 区域选择和帧提取控件
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 视频区域选择与帧提取")
                gr.Markdown("*提示：上传视频后请设置截取区域坐标和尺寸*")
                
                # 区域选择坐标
                with gr.Row():
                    x_input = gr.Number(label="X 坐标", value=120, step=1)
                    y_input = gr.Number(label="Y 坐标", value=1300, step=1)
                
                with gr.Row():
                    width_input = gr.Number(label="宽度", value=850, step=1)
                    height_input = gr.Number(label="高度", value=220, step=1)
                
                with gr.Row():
                    fps_input = gr.Slider(
                        label="截图频率 (帧/秒)", 
                        minimum=0.1, 
                        maximum=10.0, 
                        value=1.0, 
                        step=0.1
                    )
                
                # 提取按钮
                video_extract_button = gr.Button("开始截取", variant="primary")
                
                # 提取结果信息
                extract_info = gr.Textbox(
                    label="提取结果信息", 
                    value="", 
                    lines=4,
                    interactive=False
                )
                
                # 待处理截图的S3存储路径
                subtitle_s3_path = gr.Textbox(
                    label="待处理截图的S3存储路径",
                    placeholder="例如: s3://bucket-name/folder/",
                    value="s3://general-demo-3/madhouse-ads-videos/subtitle-screen-shots/french1/"
                )
                
                # 浏览按钮 - 添加variant="primary"使颜色与其他按钮一致
                browse_button = gr.Button("浏览S3图片", variant="primary")

            with gr.Column(scale=2):
                # 提取的帧展示
                extracted_frames = gr.Gallery(
                    label="提取的帧",
                    columns=3,
                    height=400,
                    elem_id="extracted_frames_gallery"
                )
                
                # 添加索引输入框，用于选择要删除的帧
                delete_frame_index = gr.Number(
                    label="输入要删除的帧索引 (从0开始)",
                    value=0,
                    step=1,
                    precision=0
                )
                
                # 添加删除按钮
                delete_frames_btn = gr.Button("删除选中的帧", variant="secondary")
                
                # 添加S3上传功能
                with gr.Row():
                    upload_s3_path = gr.Textbox(
                        label="S3 上传目录",
                        placeholder="例如: s3://bucket-name/screenshots/",
                        value="s3://general-demo-3/madhouse-ads-videos/subtitle-screen-shots/"
                    )
                    s3_upload_button = gr.Button("上传到S3", variant="primary")
                
                # 上传结果信息
                s3_upload_result = gr.Textbox(
                    label="上传结果",
                    value="",
                    lines=2,
                    interactive=False
                )
        
        # 重新组织字幕截图文字识别界面，按照用户要求调整布局
        with gr.Row():
            # 左侧列：模型选择、语言选择、提示词、处理按钮和结果显示
            with gr.Column(scale=1):
                # 模型和语言选择区域
                model_dropdown = gr.Dropdown(
                    choices=["Claude 3 Opus", "Claude 3 Sonnet", "Claude 3.5 Haiku", "Claude 3.5 Sonnet v1", "Claude 3.5 Sonnet v2", "Claude 3.7 Sonnet", "Nova Lite", "Nova Micro", "Nova Pro"],
                    label="Bedrock 模型选择",
                    value="Claude 3.7 Sonnet"
                )
                
                language_dropdown = gr.Dropdown(
                    choices=["SA", "JP", "KR", "FR", "IT", "DE", "UA", "TR"],
                    label="字幕语言",
                    value="FR"
                )
                
                # 提示词
                system_prompt = gr.Textbox(
                    label="系统提示词",
                    value="你是一个小语种字幕提取专家",
                    lines=2
                )
                
                user_prompt = gr.Textbox(
                    label="用户提示词",
                    value="请提取图片中的字幕,并用json的格式输出'原文'和翻译后的'中文',并检查原文是否有任何语法或者拼写错误，把检查结果也输出在'语法和拼写检查结果'里",
                    lines=5
                )
                
                # 处理按钮放在提示词下面
                extract_button = gr.Button("图片处理", variant="primary")
                
                # 结果显示区域放在处理按钮下面
                result_text = gr.Textbox(
                    label="识别文字结果", 
                    lines=20
                )
            
            # 右侧列：图片预览和选中的图片
            with gr.Column(scale=2):
                # 图片预览区域
                image_gallery = gr.Gallery(
                    label="S3图片预览",
                    columns=3,
                    height=350,
                    object_fit="contain"
                )
                
                # 选中的图片
                selected_image = gr.Image(
                    label="选中的图片",
                    type="pil",
                    height=350
                )
        
        # 事件处理函数
        def update_gallery(s3_path):
            images_and_metadata = list_s3_images(s3_path)
            if len(images_and_metadata) == 2:
                return images_and_metadata[0], images_and_metadata[1]
            return [], []
        
        browse_button.click(
            fn=update_gallery, 
            inputs=subtitle_s3_path, 
            outputs=[image_gallery, image_keys]
        )
        
        # 修复图片选择处理
        def handle_select(evt: gr.SelectData, s3_path, metadata_list):
            try:
                selected_index = evt.index
                if metadata_list and selected_index < len(metadata_list):
                    key = metadata_list[selected_index]["key"]
                    return select_image(key, s3_path)
                return None
            except Exception as e:
                print(f"图片选择错误: {str(e)}")
                return None
            
        image_gallery.select(fn=handle_select, inputs=[subtitle_s3_path, image_keys], outputs=selected_image)
        extract_button.click(
            fn=extract_text, 
            inputs=[selected_image, model_dropdown, language_dropdown, system_prompt, user_prompt],
            outputs=result_text
        )
        
        # 添加视频处理相关的事件处理函数
        def update_area_selection(x, y, width, height):
            """更新区域选择参数"""
            return {"x": int(x), "y": int(y), "width": int(width), "height": int(height)}
            
        def extract_frames_from_video(video_path, selection, fps):
            """从视频中提取帧"""
            if not video_path:
                return "请先上传或选择一个视频", []
            
            x = selection["x"]
            y = selection["y"]
            width = selection["width"]
            height = selection["height"]
            
            # 获取视频分辨率以验证坐标
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                # 验证坐标是否在视频范围内
                if x < 0 or y < 0 or x >= video_width or y >= video_height:
                    error_msg = f"坐标错误：(x={x}, y={y}) 不在视频范围 (0~{video_width-1}, 0~{video_height-1}) 内"
                    return error_msg, []
                
                if x + width > video_width:
                    old_width = width
                    width = video_width - x
                    info_msg = f"注意: 宽度超出范围，已自动调整为 {width} (原值: {old_width})"
                else:
                    info_msg = ""
                
                if y + height > video_height:
                    old_height = height
                    height = video_height - y
                    if info_msg:
                        info_msg += f"\n高度超出范围，已自动调整为 {height} (原值: {old_height})"
                    else:
                        info_msg = f"注意: 高度超出范围，已自动调整为 {height} (原值: {old_height})"
            else:
                info_msg = "警告: 无法读取视频信息，坐标可能不准确"
            
            # 确保宽度和高度不为0
            if width <= 0:
                width = min(200, video_width - x)
            if height <= 0:
                height = min(200, video_height - y)
                
            # 调用提取帧函数
            info, frames = extract_video_frames(video_path, x, y, width, height, fps)
            
            # 添加坐标信息到结果中
            complete_info = f"视频分辨率: {video_width}x{video_height}\n"
            complete_info += f"提取区域: 从 ({x},{y}) 开始，宽度 {width}，高度 {height}\n"
            complete_info += f"实际区域: ({x},{y}) 到 ({x+width},{y+height})\n"
            
            if info_msg:
                complete_info += f"{info_msg}\n"
                
            complete_info += info
            
            # 将提取的帧保存到全局变量中，方便后续上传
            global extracted_frame_paths
            extracted_frame_paths = frames
            
            return complete_info, frames
            
        def delete_frame_by_index(index):
            """删除指定索引的帧"""
            global extracted_frame_paths
            
            index = int(index) # 确保是整数
            
            if not extracted_frame_paths:
                return "没有可删除的帧，请先提取视频帧", extracted_frame_paths
                
            # 确保索引在有效范围内
            if index >= 0 and index < len(extracted_frame_paths):
                # 删除指定索引的帧
                deleted_path = extracted_frame_paths.pop(index)
                
                # 更新提示信息
                result = f"已删除索引为 {index} 的帧，当前剩余 {len(extracted_frame_paths)} 帧"
                
                return result, extracted_frame_paths
            else:
                return f"无效的索引: {index}。有效范围: 0-{len(extracted_frame_paths)-1}", extracted_frame_paths
                
        def upload_frames_to_s3(s3_path):
            """将提取的帧上传到S3"""
            global extracted_frame_paths
            
            if not extracted_frame_paths or len(extracted_frame_paths) == 0:
                return "没有可上传的帧，请先提取视频帧", None
                
            if not s3_path:
                return "请提供有效的S3路径", None
                
            try:
                # 处理s3://前缀
                if s3_path.startswith('s3://'):
                    s3_path = s3_path[5:]  # 移除's3://'前缀
                    
                # 解析bucket和prefix
                parts = s3_path.strip('/').split('/', 1)
                if len(parts) < 2:
                    return f"无效的S3路径: {s3_path}，格式应为 s3://bucket-name/path/", None
                    
                bucket = parts[0]
                prefix = parts[1]
                
                # 确保prefix以/结尾
                if not prefix.endswith('/'):
                    prefix += '/'
                    
                # 创建S3客户端
                s3_client = boto3.client('s3')
                
                # 创建时间戳文件夹
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                folder_prefix = f"{prefix}{timestamp}/"
                
                # 上传文件
                success_count = 0
                for idx, frame_path in enumerate(extracted_frame_paths):
                    if os.path.exists(frame_path):
                        file_name = os.path.basename(frame_path)
                        s3_key = f"{folder_prefix}{file_name}"
                        
                        s3_client.upload_file(
                            Filename=frame_path,
                            Bucket=bucket,
                            Key=s3_key
                        )
                        success_count += 1
                
                # 返回上传结果和完整的S3路径
                full_s3_path = f"s3://{bucket}/{folder_prefix}"
                return f"成功上传 {success_count} 帧到 {full_s3_path}", full_s3_path
                
            except Exception as e:
                print(f"上传到S3时出错: {str(e)}")
                return f"上传失败: {str(e)}"
                
        def handle_subtitle_video_upload(video_path):
            """处理字幕识别页面的视频上传"""
            if video_path is None:
                return None, "请上传视频文件"
            
            try:
                # 获取文件信息
                file_size = os.path.getsize(video_path)
                file_name = os.path.basename(video_path)
                
                # 获取视频分辨率
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    # 获取视频属性
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    # HTML显示
                    dimensions_html = f"""
                    <div style="padding: 15px; background-color: #f0f8ff; border: 1px solid #add8e6; border-radius: 5px; margin: 10px 0;">
                        <h3 style="color: #2c3e50; margin-top: 0;">视频分辨率信息</h3>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <div style="flex: 1; padding-right: 10px;">
                                <strong>宽度 (X轴):</strong> {width} 像素
                            </div>
                            <div style="flex: 1; padding-left: 10px;">
                                <strong>高度 (Y轴):</strong> {height} 像素
                            </div>
                        </div>
                        <div style="background-color: #e8f4f8; padding: 10px; border-radius: 3px;">
                            <strong>坐标系统:</strong> 原点(0,0)位于左上角，X向右，Y向下
                        </div>
                    </div>
                    """
                    
                    # 释放资源
                    cap.release()
                    
                    return video_path, dimensions_html
                else:
                    return video_path, "<p>无法读取视频分辨率信息</p>"
            except Exception as e:
                print(f"视频上传错误: {str(e)}")
                return None, f"<p>视频处理错误: {str(e)}</p>"
        
        # 注册视频上传事件
        upload_video.change(
            fn=handle_subtitle_video_upload,
            inputs=upload_video,
            outputs=[upload_video, video_dimensions]
        )
        
        # 注册区域选择事件
        gr.on(
            [x_input.change, y_input.change, width_input.change, height_input.change],
            fn=update_area_selection,
            inputs=[x_input, y_input, width_input, height_input],
            outputs=area_selection
        )
        
        # 注册帧提取事件
        video_extract_button.click(
            fn=extract_frames_from_video,
            inputs=[upload_video, area_selection, fps_input],
            outputs=[extract_info, extracted_frames]
        )
        
        # 注册删除帧事件
        delete_frames_btn.click(
            fn=delete_frame_by_index,
            inputs=delete_frame_index,
            outputs=[s3_upload_result, extracted_frames]
        )
        
        # 定义上传后格式化输出的函数
        def upload_and_format_result(s3_path):
            """上传帧到S3并格式化输出结果，同时更新浏览路径"""
            result_text, full_s3_path = upload_frames_to_s3(s3_path)
            
            # 如果上传成功且返回了有效的S3路径
            if full_s3_path:
                # 返回格式化的结果字符串和上传路径（用于更新浏览路径）
                return f"成功上传帧到 {full_s3_path}", full_s3_path
            else:
                # 上传失败，返回错误信息，不更新浏览路径
                return result_text, subtitle_s3_path.value
        
        # 注册S3上传事件 - 完成上传并更新浏览路径
        s3_upload_button.click(
            fn=upload_and_format_result,
            inputs=upload_s3_path,
            outputs=[s3_upload_result, subtitle_s3_path]
        )
        
        # 浏览按钮点击事件 - 手动触发浏览
        browse_button.click(
            fn=update_gallery, 
            inputs=subtitle_s3_path, 
            outputs=[image_gallery, image_keys]
        )
        
        return subtitle_ui, image_keys, subtitle_s3_path

def transcribe_video(s3_video_path, language_code):
    """使用AWS Transcribe服务转录S3视频中的语音"""
    try:
        # 创建Transcribe客户端，指定us-west-2区域
        transcribe_client = boto3.client('transcribe', region_name='us-west-2')
        
        # 生成唯一的任务名称
        job_name = f"transcribe-job-{uuid.uuid4()}"
        
        # 将语言代码映射到AWS Transcribe支持的格式
        language_mapping = {
            "SA": "ar-SA",  # 阿拉伯语
            "JP": "ja-JP",  # 日语
            "KR": "ko-KR",  # 韩语
            "FR": "fr-FR",  # 法语
            "IT": "it-IT",  # 意大利语
            "DE": "de-DE",  # 德语
            "UA": "uk-UA",  # 乌克兰语
            "TR": "tr-TR"   # 土耳其语
        }
        
        # 获取对应的AWS Transcribe语言代码
        aws_language_code = language_mapping.get(language_code, "en-US")  # 默认使用英语
        
        # 解析S3路径
        if s3_video_path.startswith('s3://'):
            s3_video_path = s3_video_path[5:]  # 移除's3://'前缀
            
        # 获取文件扩展名
        file_extension = s3_video_path.split('.')[-1].lower()
        # 映射文件扩展名到媒体格式
        media_format_mapping = {
            'mp4': 'mp4',
            'mov': 'mov',
            'avi': 'avi',
            'mkv': 'mkv',
            'wmv': 'wmv'
        }
        media_format = media_format_mapping.get(file_extension, 'mp4')  # 默认使用mp4
        
        # 解析bucket和key
        parts = s3_video_path.strip('/').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        # 启动转录任务
        # 确保MediaFileUri是正确的S3 URI格式
        if not s3_video_path.startswith('s3://'):
            s3_uri = f"s3://{bucket}/{key}"
        else:
            s3_uri = s3_video_path
            
        print(f"使用S3 URI: {s3_uri}")
        
        # 检查bucket是否存在
        s3_client = boto3.client('s3')
        try:
            s3_client.head_bucket(Bucket=bucket)
            print(f"确认S3存储桶存在: {bucket}")
            
            # 配置字幕输出
            subtitle_formats = ["srt", "vtt"]
            
            # 启动转录任务，包含字幕输出配置
            response = transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': s3_uri},
                MediaFormat=media_format,
                LanguageCode=aws_language_code,
                Subtitles={
                    'Formats': subtitle_formats,
                    'OutputStartIndex': 1
                }
            )
            
            # 返回任务名称和初始状态
            return {
                "status": "STARTED",
                "job_name": job_name,
                "message": f"已启动转录任务 {job_name}，正在处理中...\n\n任务可能需要几分钟到几小时不等，具体取决于视频长度。\n请点击'检查任务状态'按钮查看进度。"
            }
            
        except Exception as bucket_error:
            print(f"检查存储桶时出错: {str(bucket_error)}")
            return {
                "status": "ERROR",
                "message": f"转录错误: 无法访问存储桶 {bucket}，请确认存储桶存在且有访问权限"
            }
            
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"转录错误: {str(e)}"
        }

def translate_text(text, source_language_code):
    """使用AWS Translate将文本翻译成中文"""
    try:
        # 创建Translate客户端
        translate_client = boto3.client('translate', region_name='us-west-2')
        
        # 调用翻译API
        response = translate_client.translate_text(
            Text=text,
            SourceLanguageCode=source_language_code,
            TargetLanguageCode='zh'  # 中文
        )
        
        # 返回翻译结果
        return response['TranslatedText']
    except Exception as e:
        print(f"翻译错误: {str(e)}")
        return f"[翻译失败: {str(e)}]"

def parse_subtitle_file(url, format_type):
    """下载并解析字幕文件内容"""
    try:
        import requests
        response = requests.get(url)
        
        if response.status_code == 200:
            content = response.text
            
            # 根据不同格式解析内容
            if format_type == "srt":
                # 处理SRT格式，保留时间戳信息
                lines = content.split('\n')
                parsed_content = []
                
                # 用于存储当前字幕条目
                current_index = None
                current_timestamp = None
                current_text = []
                
                for line in lines:
                    line = line.strip()
                    
                    # 处理序号行
                    if line.isdigit():
                        # 如果已有内容，先保存之前的条目
                        if current_index is not None and current_timestamp is not None and current_text:
                            parsed_content.append({
                                "index": current_index,
                                "timestamp": current_timestamp,
                                "text": " ".join(current_text)
                            })
                        
                        # 开始新条目
                        current_index = line
                        current_timestamp = None
                        current_text = []
                        continue
                    
                    # 处理时间戳行
                    if '-->' in line:
                        current_timestamp = line
                        continue
                    
                    # 处理文本行
                    if line:
                        current_text.append(line)
                
                # 添加最后一个条目
                if current_index is not None and current_timestamp is not None and current_text:
                    parsed_content.append({
                        "index": current_index,
                        "timestamp": current_timestamp,
                        "text": " ".join(current_text)
                    })
                
                return {
                    "format": format_type,
                    "raw_content": content,
                    "parsed_content": parsed_content
                }
                
            elif format_type == "vtt":
                # 简单处理VTT格式，提取文本内容
                lines = content.split('\n')
                parsed_content = []
                current_text = ""
                
                # 跳过VTT头部
                start_parsing = False
                
                for line in lines:
                    if not start_parsing:
                        if line.strip() == "":
                            start_parsing = True
                        continue
                    
                    # 跳过时间戳行
                    if '-->' in line:
                        if current_text:
                            parsed_content.append(current_text.strip())
                            current_text = ""
                        continue
                    
                    # 收集文本行
                    if line.strip():
                        current_text += line + " "
                    elif current_text:
                        parsed_content.append(current_text.strip())
                        current_text = ""
                
                # 添加最后一段文本
                if current_text:
                    parsed_content.append(current_text.strip())
                
                return {
                    "format": format_type,
                    "raw_content": content,
                    "parsed_content": parsed_content
                }
            
            else:
                # 其他格式直接返回原始内容
                return {
                    "format": format_type,
                    "raw_content": content,
                    "parsed_content": [content]
                }
        else:
            return {
                "format": format_type,
                "error": f"下载失败: HTTP状态码 {response.status_code}"
            }
    
    except Exception as e:
        return {
            "format": format_type,
            "error": f"解析错误: {str(e)}"
        }

def check_transcribe_job_status(job_name):
    """检查AWS Transcribe任务状态"""
    try:
        # 创建Transcribe客户端，指定us-west-2区域
        transcribe_client = boto3.client('transcribe', region_name='us-west-2')
        
        # 获取任务状态
        response = transcribe_client.get_transcription_job(
            TranscriptionJobName=job_name
        )
        
        # 提取任务状态
        job_status = response['TranscriptionJob']['TranscriptionJobStatus']
        
        # 根据状态返回不同的信息
        if job_status == 'COMPLETED':
            # 任务完成，获取转录结果
            transcript_uri = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
            
            # 获取字幕文件URI（如果有）
            subtitle_files = {}
            subtitle_contents = {}
            
            if 'Subtitles' in response['TranscriptionJob']:
                for subtitle in response['TranscriptionJob']['Subtitles'].get('SubtitleFileUris', []):
                    # 从URI中提取格式，处理带有安全令牌的URL
                    if '?' in subtitle:
                        # 如果URL包含查询参数，从基本部分提取格式
                        base_url = subtitle.split('?')[0]
                        format_match = base_url.split('.')[-1]
                    else:
                        format_match = subtitle.split('.')[-1]
                    
                    # 检查是否是支持的格式
                    if format_match in ['srt', 'vtt']:
                        subtitle_files[format_match] = subtitle
                        # 下载并解析字幕文件内容
                        subtitle_contents[format_match] = parse_subtitle_file(subtitle, format_match)
                        print(f"已处理 {format_match} 格式字幕")
                    else:
                        print(f"不支持的字幕格式: {format_match}")
            
            # 下载转录结果
            import requests
            transcript_response = requests.get(transcript_uri)
            transcript_data = transcript_response.json()
            
            # 提取转录文本
            transcript_text = transcript_data['results']['transcripts'][0]['transcript']
            
            # 获取语言代码
            language_code = response['TranscriptionJob']['LanguageCode']
            source_language = language_code.split('-')[0]  # 提取主要语言代码，如'fr-FR'变为'fr'
            
            # 翻译转录文本
            translated_transcript = translate_text(transcript_text, source_language)
            
            # 翻译字幕内容
            if "srt" in subtitle_contents:
                srt_content = subtitle_contents["srt"]
                if "parsed_content" in srt_content and isinstance(srt_content["parsed_content"], list):
                    for item in srt_content["parsed_content"]:
                        if isinstance(item, dict) and "text" in item:
                            # 翻译字幕文本
                            item["translated_text"] = translate_text(item["text"], source_language)
            
            # 构建结果
            result = {
                "status": "COMPLETED",
                "transcript": transcript_text,
                "translated_transcript": translated_transcript,
                "subtitle_files": subtitle_files,
                "subtitle_contents": subtitle_contents,
                "message": "转录任务已完成！",
                "source_language": source_language
            }
            
            return result
            
        elif job_status == 'FAILED':
            # 任务失败
            failure_reason = response['TranscriptionJob'].get('FailureReason', '未知错误')
            return {
                "status": "FAILED",
                "message": f"转录任务失败: {failure_reason}"
            }
            
        else:
            # 任务仍在进行中
            progress = response['TranscriptionJob'].get('Progress', 0)
            return {
                "status": job_status,
                "progress": progress,
                "message": f"转录任务状态: {job_status}, 进度: {progress}%"
            }
            
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"检查任务状态时出错: {str(e)}"
        }

def extract_video_frames(video_path, x, y, width, height, fps):
    """从视频中提取指定区域的帧"""
    if not video_path or not isinstance(video_path, str):
        return "视频路径无效", []
    
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "无法打开视频文件", []
        
        # 获取视频属性
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        # 计算提取的帧间隔
        frame_interval = int(video_fps / fps) if fps > 0 else 1
        if frame_interval < 1:
            frame_interval = 1
        
        # 准备存储提取的帧
        extracted_frames = []
        temp_dir = tempfile.mkdtemp()
        
        # 按指定间隔提取帧
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按指定间隔提取帧
            if frame_count % frame_interval == 0:
                # 确保坐标不超出边界
                frame_height, frame_width = frame.shape[:2]
                crop_x = max(0, min(x, frame_width - 1))
                crop_y = max(0, min(y, frame_height - 1))
                crop_width = min(width, frame_width - crop_x)
                crop_height = min(height, frame_height - crop_y)
                
                # 裁剪区域
                if crop_width > 0 and crop_height > 0:
                    cropped = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
                    
                    # 保存裁剪的帧
                    output_path = os.path.join(temp_dir, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(output_path, cropped)
                    extracted_frames.append(output_path)
                    saved_count += 1
                
                # 限制提取的帧数量，避免过多
                if saved_count >= 50:  # 最多提取50帧
                    break
            
            frame_count += 1
        
        # 释放资源
        cap.release()
        
        # 返回结果信息和提取的帧路径
        result_info = f"成功从视频中提取了 {saved_count} 帧，帧率: {fps} fps"
        return result_info, extracted_frames
    
    except Exception as e:
        # 处理异常
        return f"提取视频帧时发生错误: {str(e)}", []

def create_video_subtitles_ui():
    """创建视频字幕获取界面"""
    with gr.Column() as video_ui:
        # 创建状态变量存储视频元数据
        video_keys = gr.State([])
        video_urls = gr.State([])
        area_selection = gr.State({"x": 120, "y": 1300, "width": 850, "height": 220})
        # 存储选中需要删除的帧索引
        selected_frames = gr.State([])
        
        # 全局变量存储提取的帧路径
        global extracted_frame_paths
        extracted_frame_paths = []
        
        # S3视频和本地上传整合在同一页面
        with gr.Row():
            with gr.Column(scale=1):
                # S3配置区域（从左侧栏加载，这里保持UI一致性）
                video_s3_path = gr.Textbox(
                    label="S3视频存储路径",
                    placeholder="例如: s3://bucket-name/videos/",
                    value="s3://general-demo-3/madhouse-ads-videos/"
                )
                
                # 浏览按钮
                browse_button = gr.Button("浏览S3视频")
                
                # 视频信息区域
                s3_video_info = gr.Textbox(
                    label="视频信息",
                    value="请先选择视频或上传本地视频...",
                    lines=5,
                    interactive=False
                )
                
                # 添加下载按钮
                download_button = gr.Button("生成下载链接", variant="primary")
                download_link = gr.HTML(visible=False)
                
                # 字幕文件链接显示 - 移动到下载链接下方
                subtitle_links = gr.HTML(
                    label="字幕文件链接",
                    value="",
                    visible=True
                )
            
            with gr.Column(scale=2):
                # 视频列表区域
                s3_video_list = gr.Dataframe(
                    headers=["文件名", "大小 (字节)"],
                    label="S3可用视频文件"
                )
                
                gr.Markdown("""
                ### S3视频说明
                
                由于S3视频文件较大，下载后观看效果更佳。点击"生成下载链接"按钮下载视频。
                """)
                
                # 添加语言选择和Transcribe按钮
                transcribe_language = gr.Dropdown(
                    choices=["SA", "JP", "KR", "FR", "IT", "DE", "UA", "TR"],
                    label="转录语言",
                    value="FR"
                )
                transcribe_button = gr.Button("Transcribe视频语音", variant="primary")
                
                # 添加任务状态检查按钮和任务ID输入框
                with gr.Row():
                    job_name_input = gr.Textbox(
                        label="转录任务ID",
                        placeholder="输入任务ID以检查状态",
                        interactive=True
                    )
                    check_status_button = gr.Button("检查任务状态", variant="secondary")
                
                # 转录结果显示
                transcribe_result = gr.Textbox(
                    label="转录结果",
                    value="",
                    lines=10,
                    interactive=False
                )
        
        # 事件处理函数
        def update_s3_video_list(s3_path):
            videos_and_metadata = list_s3_videos(s3_path)
            if len(videos_and_metadata) == 2 and videos_and_metadata[1]:
                urls = videos_and_metadata[0]
                metadata = videos_and_metadata[1]
                
                # 创建数据框展示内容
                dataframe_data = []
                for meta in metadata:
                    dataframe_data.append([
                        meta["name"],
                        meta["size"]
                    ])
                
                return dataframe_data, metadata, urls
            return [], [], []
            
        def select_s3_video(evt: gr.SelectData, metadata_list, url_list):
            try:
                selected_index = evt.index[0]  # 获取选中的行
                if metadata_list and selected_index < len(metadata_list):
                    selected_meta = metadata_list[selected_index]
                    
                    # 更新视频信息
                    info_text = f"文件名: {selected_meta['name']}\n"
                    info_text += f"路径: {selected_meta['key']}\n"
                    info_text += f"大小: {selected_meta['size']/1024/1024:.2f} MB"
                    
                    # 提示用户不直接加载视频（避免超时问题）
                    # 不返回视频URL，而是返回视频信息
                    return None, info_text + "\n\n注意：S3视频无法直接播放，请下载后观看。"
                return None, "未能找到视频信息"
            except Exception as e:
                print(f"视频选择错误: {str(e)}")
                return None, f"错误: {str(e)}"
        
        def handle_upload(video_path):
            """处理本地视频上传，并获取视频信息"""
            if video_path is None:
                return None, "请上传视频文件"
            
            try:
                # 获取文件信息
                file_size = os.path.getsize(video_path)
                file_name = os.path.basename(video_path)
                
                # 获取视频分辨率
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    # 获取视频属性
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    # 设置视频信息文本（不包含坐标系统信息，避免重复）
                    video_info = f"文件名: {file_name}\n"
                    video_info += f"大小: {file_size/1024/1024:.2f} MB\n"
                    video_info += f"分辨率: {width}x{height} 像素\n"
                    video_info += f"帧率: {fps:.2f} fps\n"
                    video_info += f"时长: {duration/60:.2f} 分钟\n"
                    
                    # 使用固定的默认值而不是根据视频大小计算
                    default_x = 120
                    default_y = 1300
                    default_width = 850
                    default_height = 220
                    
                    # HTML显示
                    dimensions_html = f"""
                    <div style="padding: 15px; background-color: #f0f8ff; border: 1px solid #add8e6; border-radius: 5px; margin: 10px 0;">
                        <h3 style="color: #2c3e50; margin-top: 0;">视频分辨率信息</h3>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <div style="flex: 1; padding-right: 10px;">
                                <strong>宽度 (X轴):</strong> {width} 像素
                            </div>
                            <div style="flex: 1; padding-left: 10px;">
                                <strong>高度 (Y轴):</strong> {height} 像素
                            </div>
                        </div>
                        <div style="background-color: #e8f4f8; padding: 10px; border-radius: 3px;">
                            <strong>坐标系统:</strong> 原点(0,0)位于左上角，X向右，Y向下
                        </div>
                    </div>
                    """
                    
                    # 释放资源
                    cap.release()
                    
                    return video_path, video_info, default_x, default_y, default_width, default_height, dimensions_html
                else:
                    return video_path, "无法读取视频信息", 0, 0, 100, 100, "<p>无法读取视频分辨率信息</p>"
            except Exception as e:
                print(f"视频上传错误: {str(e)}")
                return None, f"视频处理错误: {str(e)}", 0, 0, 100, 100, "<p>视频处理出错</p>"
                
        def update_area_selection(x, y, width, height):
            """更新区域选择参数"""
            return {"x": int(x), "y": int(y), "width": int(width), "height": int(height)}
            
        def extract_frames_from_video(video_path, selection, fps):
            """从视频中提取帧"""
            if not video_path:
                return "请先上传或选择一个视频", []
            
            x = selection["x"]
            y = selection["y"]
            width = selection["width"]
            height = selection["height"]
            
            # 获取视频分辨率以验证坐标
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                # 验证坐标是否在视频范围内
                if x < 0 or y < 0 or x >= video_width or y >= video_height:
                    error_msg = f"坐标错误：(x={x}, y={y}) 不在视频范围 (0~{video_width-1}, 0~{video_height-1}) 内"
                    return error_msg, []
                
                if x + width > video_width:
                    old_width = width
                    width = video_width - x
                    info_msg = f"注意: 宽度超出范围，已自动调整为 {width} (原值: {old_width})"
                else:
                    info_msg = ""
                
                if y + height > video_height:
                    old_height = height
                    height = video_height - y
                    if info_msg:
                        info_msg += f"\n高度超出范围，已自动调整为 {height} (原值: {old_height})"
                    else:
                        info_msg = f"注意: 高度超出范围，已自动调整为 {height} (原值: {old_height})"
            else:
                info_msg = "警告: 无法读取视频信息，坐标可能不准确"
            
            # 确保宽度和高度不为0
            if width <= 0:
                width = min(200, video_width - x)
            if height <= 0:
                height = min(200, video_height - y)
                
            # 调用提取帧函数
            info, frames = extract_video_frames(video_path, x, y, width, height, fps)
            
            # 添加坐标信息到结果中
            complete_info = f"视频分辨率: {video_width}x{video_height}\n"
            complete_info += f"提取区域: 从 ({x},{y}) 开始，宽度 {width}，高度 {height}\n"
            complete_info += f"实际区域: ({x},{y}) 到 ({x+width},{y+height})\n"
            
            if info_msg:
                complete_info += f"{info_msg}\n"
                
            complete_info += info
            
            # 将提取的帧保存到全局变量中，方便后续上传
            global extracted_frame_paths
            extracted_frame_paths = frames
            
            return complete_info, frames
            
        def upload_frames_to_s3(s3_path):
            """将提取的帧上传到S3"""
            global extracted_frame_paths
            
            if not extracted_frame_paths or len(extracted_frame_paths) == 0:
                return "没有可上传的帧，请先提取视频帧"
                
            if not s3_path:
                return "请提供有效的S3路径"
                
            try:
                # 处理s3://前缀
                if s3_path.startswith('s3://'):
                    s3_path = s3_path[5:]  # 移除's3://'前缀
                    
                # 解析bucket和prefix
                parts = s3_path.strip('/').split('/', 1)
                if len(parts) < 2:
                    return f"无效的S3路径: {s3_path}，格式应为 s3://bucket-name/path/"
                    
                bucket = parts[0]
                prefix = parts[1]
                
                # 确保prefix以/结尾
                if not prefix.endswith('/'):
                    prefix += '/'
                    
                # 创建S3客户端
                s3_client = boto3.client('s3')
                
                # 创建时间戳文件夹
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                folder_prefix = f"{prefix}{timestamp}/"
                
                # 上传文件
                success_count = 0
                for idx, frame_path in enumerate(extracted_frame_paths):
                    if os.path.exists(frame_path):
                        file_name = os.path.basename(frame_path)
                        s3_key = f"{folder_prefix}{file_name}"
                        
                        s3_client.upload_file(
                            Filename=frame_path,
                            Bucket=bucket,
                            Key=s3_key
                        )
                        success_count += 1
                
                # 根据用户反馈，不上传索引文件
                
                return f"成功上传 {success_count} 帧到 s3://{bucket}/{folder_prefix}"
                
            except Exception as e:
                print(f"上传到S3时出错: {str(e)}")
                return f"上传失败: {str(e)}"
        
        # 生成下载链接
        def generate_download_link(metadata_list, url_list, selected_index=None):
            if not metadata_list or not url_list or selected_index is None:
                return gr.HTML(visible=False)
            
            if selected_index < len(url_list):
                url = url_list[selected_index]
                meta = metadata_list[selected_index]
                html = f"""<div style="text-align:center; margin:20px;">
                <p>已生成可下载的链接（有效期24小时）:</p>
                <a href="{url}" target="_blank" download="{meta['name']}" 
                style="display:inline-block; background:#4CAF50; color:white; 
                padding:10px 20px; text-decoration:none; border-radius:5px;">
                下载 {meta['name']} ({meta['size']/1024/1024:.2f} MB)
                </a>
                </div>"""
                return gr.HTML(value=html, visible=True)
            return gr.HTML(visible=False)
        
        # 保存当前选中的行索引
        selected_row_index = gr.State(None)
        
        # 注册事件处理
        browse_button.click(
            fn=update_s3_video_list,
            inputs=video_s3_path,
            outputs=[s3_video_list, video_keys, video_urls]
        )
        
        def handle_s3_video_selection(evt: gr.SelectData, metadata_list, url_list):
            selected_index = evt.index[0]  # 获取选中的行
            info_text = select_s3_video(evt, metadata_list, url_list)
            return info_text[1], selected_index  # 只返回视频信息和选中的索引
            
        s3_video_list.select(
            fn=handle_s3_video_selection,
            inputs=[video_keys, video_urls],
            outputs=[s3_video_info, selected_row_index]
        )
        
        download_button.click(
            fn=generate_download_link,
            inputs=[video_keys, video_urls, selected_row_index],
            outputs=download_link
        )
        
        # 添加Transcribe按钮事件处理函数
        def handle_transcribe(metadata_list, selected_index, language_code):
            """处理视频转录请求"""
            if not metadata_list or selected_index is None:
                return "请先选择一个视频文件", "", False
            
            if selected_index < len(metadata_list):
                selected_meta = metadata_list[selected_index]
                video_key = selected_meta["key"]
                
                # 获取视频路径
                video_name = selected_meta["name"]
                video_key = selected_meta["key"]
                
                # 从视频存储路径中获取基础S3路径
                base_s3_path = video_s3_path.value
                if base_s3_path.startswith('s3://'):
                    base_s3_path = base_s3_path[5:]  # 移除's3://'前缀
                
                # 解析基础路径中的bucket
                base_parts = base_s3_path.strip('/').split('/', 1)
                bucket = base_parts[0]  # 从基础路径获取bucket名称
                
                # 构建完整的S3 URI
                s3_uri = f"s3://{bucket}/{video_key}"
                
                print(f"处理视频转录请求: {s3_uri}, 语言: {language_code}")
                print(f"基础S3路径: {base_s3_path}, 视频Key: {video_key}")
                
                # 调用转录函数
                result = transcribe_video(s3_uri, language_code)
                
                # 处理返回结果
                if isinstance(result, dict):
                    # 如果任务已启动，自动填充任务ID到输入框
                    if result.get("status") == "STARTED" and "job_name" in result:
                        job_name = result["job_name"]
                        return result["message"], job_name, False
                    elif result.get("status") == "ERROR":
                        return result["message"], "", False
                    else:
                        return str(result), "", False
                else:
                    # 兼容旧格式
                    return str(result), "", False
            return "无法获取所选视频信息", "", False
        
        # 添加检查任务状态的函数
        def handle_check_status(job_name):
            """检查转录任务状态"""
            if not job_name:
                return "请输入有效的任务ID", False, ""
            
            # 调用检查状态函数
            result = check_transcribe_job_status(job_name)
            
            # 添加调试信息
            print(f"检查任务状态返回结果: {result.get('status', 'UNKNOWN')}")
            if "subtitle_contents" in result:
                print(f"字幕内容类型: {type(result['subtitle_contents'])}")
                print(f"字幕内容格式: {list(result['subtitle_contents'].keys()) if isinstance(result['subtitle_contents'], dict) else 'Not a dict'}")
                if "srt" in result.get("subtitle_contents", {}):
                    srt_content = result["subtitle_contents"]["srt"]
                    print(f"SRT内容类型: {type(srt_content)}")
                    print(f"SRT内容键: {list(srt_content.keys()) if isinstance(srt_content, dict) else 'Not a dict'}")
                    if "parsed_content" in srt_content:
                        print(f"解析内容类型: {type(srt_content['parsed_content'])}")
                        print(f"解析内容长度: {len(srt_content['parsed_content'])}")
                        if srt_content['parsed_content']:
                            print(f"第一个解析项类型: {type(srt_content['parsed_content'][0])}")
                            print(f"第一个解析项内容: {srt_content['parsed_content'][0]}")
            
            # 处理返回结果
            if isinstance(result, dict):
                status = result.get("status", "UNKNOWN")
                
                # 如果任务完成，显示转录结果和字幕链接
                if status == "COMPLETED" and "transcript" in result:
                    # 构建字幕文件链接和内容HTML
                    subtitle_html = "<div style='margin-top: 15px;'>"
                    
                    # 添加字幕文件下载链接
                    subtitle_html += "<h3>字幕文件下载链接</h3>"
                    
                    if "subtitle_files" in result and result["subtitle_files"]:
                        for format_name, url in result["subtitle_files"].items():
                            subtitle_html += f"""
                            <div style="margin: 10px 0;">
                                <a href="{url}" target="_blank" download="subtitle.{format_name}" 
                                style="display:inline-block; background:#4CAF50; color:white; 
                                padding:8px 15px; text-decoration:none; border-radius:5px;">
                                下载 {format_name.upper()} 字幕文件
                                </a>
                            </div>
                            """
                        
                        # 添加字幕文件内容展示 - 只显示SRT格式
                        if "subtitle_contents" in result and result["subtitle_contents"] and "srt" in result["subtitle_contents"]:
                            subtitle_html += "<h3>字幕文件内容预览</h3>"
                            
                            # 只处理SRT格式
                            content_data = result["subtitle_contents"]["srt"]
                            subtitle_html += "<div style='margin-top: 20px;'>"
                            subtitle_html += "<h4>字幕内容预览:</h4>"
                            
                            if "error" in content_data:
                                subtitle_html += f"<p style='color: red;'>{content_data['error']}</p>"
                            else:
                                # 显示解析后的内容
                                subtitle_html += "<div style='max-height: 300px; overflow-y: auto; padding: 10px; background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 5px;'>"
                                
                                if "parsed_content" in content_data and content_data["parsed_content"]:
                                    for i, item in enumerate(content_data["parsed_content"]):
                                        if isinstance(item, dict) and "index" in item and "timestamp" in item and "text" in item:
                                            # 新格式，包含时间戳
                                            subtitle_html += f"""
                                            <div style='margin: 10px 0; padding: 5px; border-bottom: 1px solid #eee;'>
                                                <div style='color: #666; font-size: 0.9em;'>{item["timestamp"]}</div>
                                                <div style='margin-top: 3px;'>
                                                    <div style='margin-bottom: 5px;'><strong>原文：</strong>{item["text"]}</div>
                                                    <div style='color: #0066cc;'><strong>中文翻译：</strong>{item.get("translated_text", "无翻译")}</div>
                                                </div>
                                            </div>
                                            """
                                        else:
                                            # 兼容旧格式
                                            subtitle_html += f"<p style='margin: 5px 0;'>{i+1}. {item}</p>"
                                else:
                                    subtitle_html += "<p>无法解析字幕内容</p>"
                                    
                                subtitle_html += "</div>"
                            
                            subtitle_html += "</div>"
                    else:
                        subtitle_html += "<p>没有可用的字幕文件</p>"
                    
                    subtitle_html += "</div>"
                    
                    # 返回转录文本（包括原文和中文翻译）和字幕链接
                    transcript_display = f"{result['message']}\n\n"
                    transcript_display += "转录文本(原文):\n"
                    transcript_display += f"{result['transcript']}\n\n"
                    transcript_display += "转录文本(中文翻译):\n"
                    transcript_display += f"{result.get('translated_transcript', '无翻译')}"
                    
                    return transcript_display, True, subtitle_html
                elif status == "FAILED":
                    return result["message"], False, ""
                else:
                    # 任务仍在进行中
                    progress_info = f"进度: {result.get('progress', 0)}%" if "progress" in result else ""
                    return f"{result['message']}\n{progress_info}", False, ""
            else:
                # 兼容旧格式
                return str(result), False, ""
        
        # 注册Transcribe按钮事件
        transcribe_button.click(
            fn=handle_transcribe,
            inputs=[video_keys, selected_row_index, transcribe_language],
            outputs=[transcribe_result, job_name_input, subtitle_links]
        )
        
        # 注册检查状态按钮事件
        check_status_button.click(
            fn=handle_check_status,
            inputs=job_name_input,
            outputs=[transcribe_result, subtitle_links, subtitle_links]  # 第三个参数是HTML内容
        )
        
        # 修改为只返回视频信息，不设置坐标
        def handle_upload_simplified(video_path):
            """处理本地视频上传，简化版本只返回视频信息"""
            if video_path is None:
                return None, "请上传视频文件"
            
            try:
                # 获取文件信息
                file_size = os.path.getsize(video_path)
                file_name = os.path.basename(video_path)
                
                # 获取视频分辨率
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    # 获取视频属性
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    # 设置视频信息文本
                    video_info = f"文件名: {file_name}\n"
                    video_info += f"大小: {file_size/1024/1024:.2f} MB\n"
                    video_info += f"分辨率: {width}x{height} 像素\n"
                    video_info += f"帧率: {fps:.2f} fps\n"
                    video_info += f"时长: {duration/60:.2f} 分钟\n"
                    
                    # 释放资源
                    cap.release()
                    
                    return video_path, video_info
                else:
                    return video_path, "无法读取视频信息"
            except Exception as e:
                print(f"视频上传错误: {str(e)}")
                return None, f"视频处理错误: {str(e)}"
                
        # 删除对不存在的upload_video的引用
        
        # 删除对不存在的坐标输入控件的引用
        
        # 删除对不存在的extract_button和upload_video的引用
        
        # 删除指定索引的帧
        def delete_frame_by_index(index):
            global extracted_frame_paths
            
            index = int(index) # 确保是整数
            
            if not extracted_frame_paths:
                return "没有可删除的帧，请先提取视频帧", extracted_frame_paths
                
            # 确保索引在有效范围内
            if index >= 0 and index < len(extracted_frame_paths):
                # 删除指定索引的帧
                deleted_path = extracted_frame_paths.pop(index)
                
                # 更新提示信息
                result = f"已删除索引为 {index} 的帧，当前剩余 {len(extracted_frame_paths)} 帧"
                
                return result, extracted_frame_paths
            else:
                return f"无效的索引: {index}。有效范围: 0-{len(extracted_frame_paths)-1}", extracted_frame_paths
        
        # 删除对不存在的delete_frames_btn的引用
        
        # 删除对不存在的s3_upload_button的引用
        
        # 创建一个空的上传路径变量，以保持返回值结构一致
        upload_s3_path_placeholder = gr.State("s3://general-demo-3/madhouse-ads-videos/subtitle-screen-shots/")
        return video_ui, video_s3_path, upload_s3_path_placeholder

def create_app():
    """创建主应用"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 视频字幕工具")
        
        # 创建共享状态变量
        image_keys = gr.State([])
        current_tab = gr.State("video")  # 默认为视频字幕获取标签页
        
        # 创建全局S3路径状态变量
        s3_videos_path_state = gr.State("s3://general-demo-3/madhouse-ads-videos/")
        s3_screenshots_path_state = gr.State("s3://general-demo-3/madhouse-ads-videos/subtitle-screen-shots/french1/")
        s3_upload_path_state = gr.State("s3://general-demo-3/madhouse-ads-videos/subtitle-screen-shots/")
        
        with gr.Row():
            # 左侧导航栏 (1/5宽度)
            with gr.Column(scale=1):
                gr.Markdown("### 导航菜单")
                with gr.Column():
                    video_btn = gr.Button("视频字幕获取", variant="primary")
                    subtitle_btn = gr.Button("字幕截图文字识别", variant="secondary")
                
                # S3配置区域
                gr.Markdown("### S3路径设置", elem_id="s3-settings-title")
                
                # 各种S3路径输入框
                s3_videos_path = gr.Textbox(
                    label="视频存储路径",
                    placeholder="例如: s3://bucket-name/videos/",
                    value="s3://general-demo-3/madhouse-ads-videos/",
                )
                
                s3_upload_path = gr.Textbox(
                    label="上传目录路径",
                    placeholder="例如: s3://bucket-name/uploads/",
                    value="s3://general-demo-3/madhouse-ads-videos/subtitle-screen-shots/"
                )
                
                # 加载设置按钮
                load_settings_btn = gr.Button("加载设置", variant="primary")
            
            # 右侧内容区 (4/5宽度)
            with gr.Column(scale=4):
                # 创建内容区容器 - 默认显示视频字幕获取标签页
                subtitle_container = gr.Column(visible=False)
                video_container = gr.Column(visible=True)
                
                # 向容器中添加UI组件
                with subtitle_container:
                    subtitle_ui, image_keys, subtitle_s3_path = create_subtitle_recognition_ui()
                
                with video_container:
                    video_ui, video_s3_path, upload_s3_path = create_video_subtitles_ui()
        
        # 导航切换事件处理
        def show_subtitle_tab():
            return {
                subtitle_container: gr.Column(visible=True),
                video_container: gr.Column(visible=False),
                subtitle_btn: gr.Button(variant="primary"),
                video_btn: gr.Button(variant="secondary")
            }
            
        def show_video_tab():
            return {
                subtitle_container: gr.Column(visible=False),
                video_container: gr.Column(visible=True),
                subtitle_btn: gr.Button(variant="secondary"),
                video_btn: gr.Button(variant="primary")
            }
            
        # 加载S3设置函数
        def load_s3_settings(videos_path, upload_path):
            # 更新状态变量
            return {
                # 更新字幕识别UI中的路径 - 使用上传路径作为浏览路径
                subtitle_s3_path: gr.Textbox(value=upload_path),
                # 更新视频字幕UI中的路径
                video_s3_path: gr.Textbox(value=videos_path),
                # 更新上传路径
                upload_s3_path: gr.Textbox(value=upload_path),
                # 更新状态变量
                s3_videos_path_state: videos_path,
                s3_upload_path_state: upload_path
            }
        
        # 注册导航按钮点击事件
        subtitle_btn.click(
            fn=show_subtitle_tab,
            inputs=None,
            outputs=[subtitle_container, video_container, subtitle_btn, video_btn]
        )
        
        video_btn.click(
            fn=show_video_tab,
            inputs=None,
            outputs=[subtitle_container, video_container, subtitle_btn, video_btn]
        )
        
        # 注册加载设置按钮事件
        load_settings_btn.click(
            fn=load_s3_settings,
            inputs=[s3_videos_path, s3_upload_path],
            outputs=[
                subtitle_s3_path,                          # 字幕识别UI中的路径
                video_s3_path,                             # 视频字幕UI中的视频路径
                upload_s3_path,                            # 上传路径
                s3_videos_path_state,                      # 状态变量
                s3_upload_path_state                       # 上传路径状态变量
            ]
        )
        
    return demo

if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)

import gradio as gr
import boto3
from PIL import Image
import io
import base64
import json
import os

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
        
        with gr.Row():
            with gr.Column(scale=1):
                # S3配置区域
                s3_path = gr.Textbox(
                    label="S3存储路径",
                    placeholder="例如: s3://bucket-name/folder/",
                    value="s3://general-demo-3/madhouse-ads-videos/subtitle-screen-shots/french1/"
                )
                
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
                
                # 浏览按钮
                browse_button = gr.Button("浏览S3图片")
                
                # 系统和用户提示词
                system_prompt = gr.Textbox(
                    label="系统提示词",
                    value="你是一个小语种字幕提取专家",
                    lines=2
                )
                
                user_prompt = gr.Textbox(
                    label="用户提示词",
                    value="请提取图片中的字幕，并用json的格式输出'原文'和翻译后的'中文'",
                    lines=3
                )
                
                # 结果显示区域
                result_text = gr.Textbox(label="识别文字结果", lines=8)
                
            with gr.Column(scale=2):
                # 图片预览和选择区域
                with gr.Row():
                    image_gallery = gr.Gallery(
                        label="S3图片预览",
                        columns=3,
                        height=400,
                        object_fit="contain"
                    )
                    
                with gr.Row():
                    selected_image = gr.Image(
                        label="选中的图片",
                        type="pil",
                        height=400
                    )
                
                extract_button = gr.Button("图片处理", variant="primary")
        
        # 事件处理函数
        def update_gallery(s3_path):
            images_and_metadata = list_s3_images(s3_path)
            if len(images_and_metadata) == 2:
                return images_and_metadata[0], images_and_metadata[1]
            return [], []
        
        browse_button.click(
            fn=update_gallery, 
            inputs=s3_path, 
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
            
        image_gallery.select(fn=handle_select, inputs=[s3_path, image_keys], outputs=selected_image)
        extract_button.click(
            fn=extract_text, 
            inputs=[selected_image, model_dropdown, language_dropdown, system_prompt, user_prompt],
            outputs=result_text
        )
        
        return subtitle_ui, image_keys

def create_video_subtitles_ui():
    """创建视频字幕获取界面"""
    with gr.Column() as video_ui:
        # 创建状态变量存储视频元数据
        video_keys = gr.State([])
        video_urls = gr.State([])
        
        # S3视频和本地上传整合在同一页面
        with gr.Row():
            with gr.Column(scale=1):
                # S3配置区域
                s3_path = gr.Textbox(
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

        # 视频上传与播放区域合并
        with gr.Row():
            # 文件上传和播放组件合并
            upload_video = gr.Video(
                label="上传或播放视频（可直接拖拽上传本地视频）",
                interactive=True,
                height=400
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
            """处理本地视频上传"""
            if video_path is None:
                return None
            try:
                # 获取文件信息
                file_size = os.path.getsize(video_path)
                file_name = os.path.basename(video_path)
                return video_path
            except Exception as e:
                print(f"视频上传错误: {str(e)}")
                return None
        
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
            inputs=s3_path,
            outputs=[s3_video_list, video_keys, video_urls]
        )
        
        def handle_s3_video_selection(evt: gr.SelectData, metadata_list, url_list):
            selected_index = evt.index[0]  # 获取选中的行
            return select_s3_video(evt, metadata_list, url_list) + (selected_index,)
            
        s3_video_list.select(
            fn=handle_s3_video_selection,
            inputs=[video_keys, video_urls],
            outputs=[upload_video, s3_video_info, selected_row_index]
        )
        
        download_button.click(
            fn=generate_download_link,
            inputs=[video_keys, video_urls, selected_row_index],
            outputs=download_link
        )
        
        upload_video.change(
            fn=handle_upload,
            inputs=upload_video,
            outputs=upload_video
        )
        
        return video_ui

def create_app():
    """创建主应用"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 视频字幕工具")
        
        # 创建共享状态变量
        image_keys = gr.State([])
        current_tab = gr.State("subtitle")
        
        with gr.Row():
            # 左侧导航栏 (1/5宽度)
            with gr.Column(scale=1):
                gr.Markdown("### 导航菜单")
                with gr.Column():
                    video_btn = gr.Button("视频字幕获取", variant="secondary")
                    subtitle_btn = gr.Button("字幕截图文字识别", variant="primary")
            
            # 右侧内容区 (4/5宽度)
            with gr.Column(scale=4):
                # 创建内容区容器
                subtitle_container = gr.Column(visible=True)
                video_container = gr.Column(visible=False)
                
                # 向容器中添加UI组件
                with subtitle_container:
                    subtitle_ui, image_keys = create_subtitle_recognition_ui()
                
                with video_container:
                    video_ui = create_video_subtitles_ui()
        
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
        
    return demo

if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)

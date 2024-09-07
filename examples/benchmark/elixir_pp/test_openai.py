import openai

# 函数：调用 OpenAI API
def call_openai(api_key, base_url, model="gpt-4o-mini", prompt="Hello, how are you?", temperature=0.7):
    # 配置 API key 和基地址
    openai.api_key = api_key
    openai.api_base = base_url  # 设置自定义的 OpenAI 基地址

    # 调用 OpenAI API 生成文本
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        # 返回 API 的回复
        return response.choices[0].message['content']
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# 示例：传入API key和基地址
api_key = "sk-A1bZdzgU1EKQ2gjn0424C63f6e9243D1B9FfAf8515A083C8"  # 在此替换为你的 OpenAI API 密钥
base_url = "https://api.openai99.top/v1"  # 或者是自定义的 OpenAI API 基地址
prompt = "Who are you?"

# 调用函数并打印结果
result = call_openai(api_key, base_url, prompt=prompt)
print(f"OpenAI Response: {result}")

from playwright.sync_api import sync_playwright

for browser in "firefox chromium".split():
    with sync_playwright() as p:
        # 指定用户数据目录的路径
        user_data_dir = f"./data/{browser}" 
    
        # 使用 launch_persistent_context 启动
        context = getattr(p, browser).launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False  # 设为 False 以便观察浏览器
        )
        
        # 从持久化上下文创建一个新页面
        page = context.new_page()
        page.goto("https://example.com")
        
        # 你的自动化操作...
        
        # 关闭上下文，浏览器会自动关闭
        context.close()
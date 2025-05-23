# AI Box Search Agent V4

### Version
- v3.0
    - 采用planner、searcher、reader、reasoner四个模块，searchagent格式为muli-agent交互的格式。
    - searcher使用标准tool call 格式
    - searcher支持多轮。
- v3.1
    - searchagent支持与用户的多轮交互
- v4.0 
    - 采用planner、searcher、reader三个模块，searchagent格式为单agent function call的格式。
    - planner被包装为执行搜索的agent（即searchagent）使用标准tool call格式
    - searcher被包装在tool中
- v5.0
    - 延续v3格式，去掉reasoner，searchagent由planner、searcher和reader组成。
    - 优化prompt
    
### Run
在本项目中，所有的环境变量都通过 .env 文件进行管理。该文件包含了项目运行所需的敏感信息、配置信息及默认值。本项目使用.env文件管理环境变量和相关配置，确保根据你的开发环境和部署需求，正确配置这些变量。
1. 创建.env文件
为了避免泄露敏感信息，确保 .env 文件没有被添加到版本控制系统中。
```
cp .env.local .env
```
2. 配置 API KEY
在 .env 文件中，您需要设置不同的 API 密钥。这些密钥将被项目用于与外部服务进行交互。
3. 仅运行后端
```
cd WebRAG/searchagent
python -m run.py
```
3. 启动前端
```
cd WebRAG
streamlit run client.py --server.port 8888
```

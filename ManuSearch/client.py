import argparse, json, asyncio, re, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from searchagent.agent.agent import AgentInterface
from searchagent.utils.utils import *


def display_chat(chat):
    with st.chat_message("user", avatar=chat["avatar"]):
        st.markdown(chat["input"], unsafe_allow_html=True)
    column_list = st.columns(len(chat["response"]))
    for i, (column, response) in enumerate(zip(column_list, chat["response"])):
        if response['text']:
            with column:
                with st.chat_message("assistant", avatar=response["avatar"]):
                    if 'think' in response['text'] and response['text']['think'].strip():
                        with st.expander("Think", expanded=True):
                            st.markdown(response['text'].get('think', ''), unsafe_allow_html=True)
                    if 'concise_answer' in response['text'] and response['text']['concise_answer'].strip():
                        pattern = r'\[\[\d+\]\]'
                        content_concise = re.sub(pattern, '', response['text']['concise_answer'].strip())
                        st.markdown("#### Concise answer:")
                        st.markdown(content_concise, unsafe_allow_html=True)
                    if 'detailed_answer' in response['text'] and response['text']['detailed_answer'].strip():
                        st.markdown("#### Detailed answer:")
                        st.markdown(response['text']['detailed_answer'].strip(), unsafe_allow_html=True)
            
            
# input: str output: str in quote
def parse_to_quote(str):
    lines = str.split('\n')
    updated_lines = []
    for line in lines:
        updated_lines.append(f"> {line}")
    return '\n'.join(updated_lines)

def replace_references(text, ref_dict):
    def replace_match(match):
        index = match.group(1)
        k_type=type(list(ref_dict.keys())[0])
        index=k_type(index)
        if index in ref_dict:
            url = ref_dict[index]['url']
            if url:
                return f'<sup><a href="{url}" target="_blank">[{index}]</a></sup>'
            else:
                return ''
        else:
            return ''
    
    if not text:
        return ''
    
    if not ref_dict:
        text = re.sub(r'\[\[\]\]', '', text)
        text = re.sub(r'\[\[(\d+)\]\]', '', text)
        return text
    
    if isinstance(text, list):
        text = '\n'.join(text)
        
    text = re.sub(r'\[\[\]\]', '', text)
    res = re.sub(r'\[\[(\d+)\]\]', replace_match, text)
    return res

async def update_widget(widget, think_process_with_ref, content_detailed, content_concise):
    with widget.container():
        if think_process_with_ref:
            with st.expander("Think", expanded=True):
                st.markdown(think_process_with_ref, unsafe_allow_html=True)
        if content_concise:
            pattern = r'\[\[\d+\]\]'
            content_concise = re.sub(pattern, '', content_concise)
            st.markdown("#### Concise answer:")
            st.markdown(content_concise, unsafe_allow_html=True)
        if content_detailed:
            st.markdown("#### Detailed answer:")
            st.markdown(content_detailed, unsafe_allow_html=True)
            
def parse_webpage_info(webpages, sidebar_container, status_text):
    with sidebar_container.container():
        sidebar_container.empty()
        for page in webpages:
            status_text.write(f'## Read {len(webpages)} pages')
            with st.sidebar.expander(page['title']):
                st.write(f"[{page['title']}]({page['url']})") # link to the page
                st.write(page['summ'])
    return sidebar_container, status_text

def parse_new_searcher_response(substatus, content, use_en, ref2url):
    output = '\n'
    if substatus == 'final_answer' or substatus == 'model_response': # str 
        output += replace_references(re.sub(r'#+', '#####', content), ref2url)
    elif substatus == 'GoogleSearch':# content: subquery, list[str]
        subquery_str = " ".join(f"`{item}`" for item in content)
        output += f"{subquery_str}\n\n"
    else:
        resp_json = parse_resp_to_json(content) if not isinstance(content, dict) else content
        if not resp_json:
            return output
            
        if substatus == 'VisitPage': # dict{url, title}
            output += 'Browsing webpage:\n' if use_en else '正在浏览网页:\n'
            page_list = ''
            for url, title in resp_json.items():
                page_list += f'- [{title}]({url})\n'
            output += page_list
        
    return output

async def fetch_response_iterative(widget, prompt, response_dict, solve_method, model_mode, history):   
    
    def parse_planer_response(plan_proces_dict):  
        md_output="\n"
        if plan_proces_dict:
            if isinstance(plan_proces_dict, list):
                plan_proces_dict = plan_proces_dict[0]
            
            if plan_proces_dict.get('actions', '') == 'final':
                md_output += f"**{plan_proces_dict.get('evaluation_previous_goal', '').strip()}**\n\n"
            else:
                for key, value in plan_proces_dict.items():
                    if not value:
                        continue
                    if key == 'think':
                        md_output += f"**{value.strip()}**\n\n"
                    elif key == 'content':
                        md_output += f'- **{value}**\n\n'
        return md_output 

    def parse_previous_response(finished_line):
        if not finished_line:
            return ""

        result = finished_line[0]['content']
        for i in range(1, len(finished_line)):
            current_item = finished_line[i]
            previous_item = finished_line[i - 1]
            if current_item['from'] == previous_item['from']:
                separator = '\n'
            else:
                separator = '\n\n'
            result += separator + current_item['content'].strip()

        return result
    
    # process history
    history_copy = history.copy()
    history_qa = []
    for item in history_copy:
        if isinstance(item, tuple) and item[1]: # complete QA
            history_qa.append(item[0]+'\n'+(item[1].get('concise_answer', '') or item[1].get('detailed_answer', '')))
            if len(history_qa) > 5: # 简单滑动窗口
                history_qa.pop(0)
    history_qa = '\n'.join(history_qa)
    
    try:
        finished_line = []
        SearchAgent = AgentInterface()
        async for step, use_en in SearchAgent.get_answer(prompt, solve_method=solve_method, deep_reasoning=model_mode, history=history_qa):
            if step['status'] == 'planning':
                plan_process = parse_planer_response(step['plan'])
                current = {'content': plan_process, 'from': 'planner'}
                if not finished_line or finished_line[-1]['from'] != 'planner':
                    finished_line.append(current)
                else:
                    finished_line[-1] = current
                all_think = parse_previous_response(finished_line)
                all_think = all_think.strip() or 'No subproblems to be extracted.\n'
                response_dict['text']['think'] = all_think
                await update_widget(widget, response_dict['text']['think'], '', '')
                
            elif step['status'] == 'searching':
                think_process = parse_new_searcher_response(substatus=step['substatus'], content=step['tool_return'], use_en=use_en, ref2url=step['ref2url'])
                current = {'content': parse_to_quote(think_process), 'from': 'reader', 'substatus': step['substatus']}
                if not finished_line or finished_line[-1]['from'] != 'reader' or finished_line[-1]['substatus'] != step['substatus']:
                    finished_line.append(current)
                else:
                    finished_line[-1] = current
                all_think = parse_previous_response(finished_line)
                all_think = all_think.strip() or 'No subproblems to be extracted.\n'
                response_dict['text']['think'] = all_think
                await update_widget(widget, response_dict['text']['think'], '', '')
                
            elif step['status'] == 'reasoning':
                reasoner_resp = parse_resp_to_json(step['final_resp'].get("content", ''))
                if not isinstance(reasoner_resp, dict):
                    continue
                if reasoner_resp:
                    response_dict['text']['concise_answer'] = replace_references(reasoner_resp.get('concise_answer', ''), step['ref2url'])
                    response_dict['text']['detailed_answer'] = replace_references(reasoner_resp.get('detailed_answer', '') or step['final_resp'].get('detailed_answer', ''), step['ref2url'])
                    await update_widget(widget, response_dict['text'].get('think', ''), response_dict['text'].get('detailed_answer', ''), response_dict['text'].get('concise_answer', ''))
                
            elif step['status'] == 'webpages':
                status_text = st.sidebar.empty()
                sidebar_container = st.sidebar.empty()
                webpages = step['content'].values()
                sidebar_container, status_text = parse_webpage_info(webpages=webpages, sidebar_container=sidebar_container, status_text=status_text)

    except Exception as e:
        print(f"Error fetching response: {e}")
        error_type = type(e).__name__
        widget.markdown(f"发生错误:\n{error_type}\n{str(e)}\n请求失败，请刷新并重试!", unsafe_allow_html=True)

        
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="./client.json",
        help="path to configs",
    )
    parser.add_argument(
        "--user_avatar",
        "-a",
        type=str,
        default="assets/user.png",
        help="path to user avatar",
    )
    return parser.parse_known_args()


def load_model_configs(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def setup_sidebar(model_config_list):
    """
    设置侧边栏，用于配置模型生成参数。

    Args:
        model_config_list (list): 模型配置列表。

    Returns:
        tuple: (kwargs_list, visibility_list)，分别存储每个模型的生成参数和可见性设置。
    """
    kwargs_list = []
    visibility_list = []
    for model_config in model_config_list:
        kwargs = {}
        for key, key_config in model_config["generate_parameter"].items():
            if not isinstance(key_config, dict):
                kwargs[key] = key_config
                continue
            kwargs[key] = 0
        kwargs_list.append(kwargs)
        visibility = True
        visibility_list.append(visibility)
    return kwargs_list, visibility_list


def initialize_session_state():
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

def display_chat_history(chat_container):
    with chat_container:
        for chat in st.session_state["chat"]:
            display_chat(chat)
            
def handle_user_input(user_input, args, chat_container, model_config_list, kwargs_list, visibility_list, solve_method, deep_reasoning):
    """
    处理用户输入，调用模型生成响应，并更新聊天记录。

    Args:
        user_input (str): 用户输入内容。
        args (argparse.Namespace): 命令行参数。
        chat_container (streamlit.container): 用于显示聊天记录的容器。
        model_config_list (list): 模型配置列表。
        kwargs_list (list): 每个模型的生成参数列表。
        visibility_list (list): 每个模型的可见性设置列表。
        interaction_mode (str): 交互方式 ("sequential" 或 "iterative")。
    """
    if user_input:
        print(f"User input received: {user_input}")
        chat = {"input": user_input, "avatar": args.user_avatar, "response": []}
        with chat_container:
            with st.chat_message("user", avatar=args.user_avatar):
                st.markdown(user_input, unsafe_allow_html=True)
            column_list = st.columns(sum(visibility_list))

            loop = asyncio.new_event_loop()
            waiting_list = []
            model_idx = 0
            for model_config, kwargs, visibility in zip(model_config_list, kwargs_list, visibility_list):
                if not visibility:
                    continue
                column = column_list[model_idx]
                with column:
                    with st.chat_message("assistant", avatar=model_config["avatar"]):
                        widget = st.empty()
                history = []
                for history_chat in st.session_state["chat"]:
                    history.append(
                        (
                            history_chat["input"],
                            history_chat["response"][model_idx]["text"],
                        )
                    )
                chat["response"].append(
                    {
                        "model": 'Search Agent',
                        "text": {},
                        "avatar": model_config["avatar"],
                    }
                )
                if solve_method == 'Auto':
                    waiting_list.append(
                        fetch_response_iterative(widget, user_input, chat["response"][-1], solve_method, deep_reasoning, history)
                    )

                model_idx += 1

            loop.run_until_complete(asyncio.wait(waiting_list))
            st.session_state["chat"].append(chat)
            
            # log chat response
            for idx, chat_response in enumerate(chat["response"]):
                print(f"Model: {chat_response['model']}, Response: {chat_response['text']}")


def main():
    if "toggle_state" not in st.session_state:
        st.session_state.toggle_state = False
    
    args, _ = parse_arguments()

    model_config_list = load_model_configs(args.config)

    TITLE = "AI Box Search Agent V5.1"
    st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

    st.title(TITLE)
    initialize_session_state()

    kwargs_list, visibility_list = setup_sidebar(model_config_list)

    chat_container = st.container()
    display_chat_history(chat_container)

    with stylable_container(
        key="bottom_content",
        css_styles=""" 
            {
                position: fixed;
                bottom: 0px;
                justify-content: space-between;
                background-color: #ffffff;
                padding: 0px 30px 50px 30px;
                margin-top: 50px;
            }
        """,
    ):
        with st.container():
            col1,  col2, col3, col4 = st.columns([5,4,4, 14.5])
            if "toggle_state" not in st.session_state:
                st.session_state.toggle_state = False
            with col1:
                solve_method = st.selectbox(
                    "",
                    ["Auto"],
                )
            with col2:
                toggle_button = st.button(" Deep Reasoning ", key="unique_toggle_button_key", type="primary")
            with col3:
                new_conv_button = st.button("New Conversation", key="unique_new_conv_button", type="secondary")
            with col4:
                st.empty()
            user_input = st.chat_input("")

    if toggle_button:
        st.session_state.toggle_state = not st.session_state.toggle_state

    if new_conv_button:
        st.session_state.clear()
        st.rerun()

    if st.session_state.toggle_state:
        button_color_deep = "#b8f2ff"
        text_color_deep = "#4a73f4"
    else:
        button_color_deep = "#f0f2f6"
        text_color_deep = "black"
    
    st.markdown(
        f"""
        <style>
            div.stButton > button {{
                flex: 1; /* 使按钮自适应列宽 */
                font-size: 16px; /* 文字大小 */
                padding: 10px; /* 按钮内边距 */
                border: none; /* 移除边框 */
                border-radius: 5px; /* 圆角按钮 */
                margin-top: 28px; /* 顶部间距 */
                margin-bottom: 1px; /* 底部间距 */
                height: 40px;
                width: 100%
            }}
            div.stButton > button[kind="primary"] {{
                background-color: {button_color_deep};
                color: {text_color_deep};
            }}
            div.stButton > button[kind="secondary"] {{
                background-color: #f0f2f6;
                color: black;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    handle_user_input(user_input, args, chat_container, model_config_list, kwargs_list, visibility_list, solve_method, st.session_state.toggle_state)


if __name__ == "__main__":


    main()

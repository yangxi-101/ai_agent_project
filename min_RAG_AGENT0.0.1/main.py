from agent import agent_exe

def main():
    print("=== 售后客服 ===‘输入exit 退出’")

    while True:
        try:
            user_input = input("用户：").strip() # 标准防御写法
        except (EOFError, KeyboardInterrupt):
            print('\n退出')
            break

        if not user_input:
            continue
        if user_input.lower() in {'exit','quit'}:
            break

        if not user_input:
            continue
        response = agent_exe.invoke({'input':user_input})

        print('Agent:',response["output"])

if __name__ == "__main__":
    main()


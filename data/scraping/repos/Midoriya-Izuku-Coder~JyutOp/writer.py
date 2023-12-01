import flet as ft
import openai
openai.api_key = input("OPENAI_API_KEY: ")
messages = [{"role": "system", "content": "你是一個大型語言模型，具有寫很好的粵劇劇本歌詞的能力，可以寫很好的粵劇歌詞，以歌詞為對白，請使用繁體字以及文言文，亦必須要有對仗、音韻之美。"}]
mb_def_style = ft.ButtonStyle(
    color={
        ft.MaterialState.HOVERED: ft.colors.WHITE,
        ft.MaterialState.FOCUSED: ft.colors.BLUE,
        ft.MaterialState.DEFAULT: ft.colors.BLACK,
    },
    bgcolor={
        ft.MaterialState.HOVERED: "#807095",
        ft.MaterialState.FOCUSED: "#807095",
        ft.MaterialState.DEFAULT: "#F9F9F8",
    },
    shape=ft.buttons.RoundedRectangleBorder(radius=0),
    elevation=1
)
mb_mod_style = ft.ButtonStyle(
    color={
        ft.MaterialState.HOVERED: ft.colors.WHITE,
        ft.MaterialState.FOCUSED: ft.colors.BLUE,
        ft.MaterialState.DEFAULT: ft.colors.WHITE,
    },
    bgcolor={
        ft.MaterialState.HOVERED: "#807095",
        ft.MaterialState.FOCUSED: "#807095",
        ft.MaterialState.DEFAULT: "#504278",
    },
    shape=ft.buttons.RoundedRectangleBorder(radius=0),
    elevation=0
)
sub_but_style = ft.ButtonStyle(
    color={
        ft.MaterialState.HOVERED: ft.colors.WHITE,
        ft.MaterialState.FOCUSED: ft.colors.BLUE,
        ft.MaterialState.DEFAULT: ft.colors.WHITE,
    },
    bgcolor={
        ft.MaterialState.HOVERED: "green",
        ft.MaterialState.FOCUSED: "green",
        ft.MaterialState.DEFAULT: "#30337530",
    },
    shape=ft.buttons.RoundedRectangleBorder(radius=4),
    elevation=1
)
msg_p = ""
def ret(page):
    def sub_prompt(e):
        global msg_p, messages
        e.text="Loading... (This may take a while)"
        e.style=ft.ButtonStyle(
            color={
                ft.MaterialState.HOVERED: ft.colors.WHITE,
                ft.MaterialState.FOCUSED: ft.colors.WHITE,
                ft.MaterialState.DEFAULT: ft.colors.WHITE,
            },
            bgcolor={
                ft.MaterialState.HOVERED: "green",
                ft.MaterialState.FOCUSED: "green",
                ft.MaterialState.DEFAULT: "green",
            },
            shape=ft.buttons.RoundedRectangleBorder(radius=0),
            elevation=0
        )
        user_prompt = "請按以下文字寫一個粵劇劇本歌詞： {}。".format(msg_p)
        print(user_prompt)
        messages.append({"role": "user", "content": user_prompt})
        print(messages)
        apireturn  = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        print(apireturn['choices'][0]['message']['content'])
        oa_rt.value = apireturn['choices'][0]['message']['content']
        page.update()
        e.style=sub_but_style
        page.update()
        pass
    def on_change(e):
        global msg_p
        msg_p = e.control.value
        print(msg_p)
    oa_title = ft.Text(
        "Opera AI\nScript Writing Assistant",
        size=50,
        color=ft.colors.WHITE,
        text_align=ft.TextAlign.CENTER
    )
    oa_sub = ft.Text(
        "Enter an unfinished script or a brief prompt for a Chinese Opera script and watch the magic happen",
        size=15,
        color=ft.colors.WHITE,
        text_align=ft.TextAlign.CENTER
    )
    oa_tb = ft.TextField(
        width=page.width-100,
        label="Prompt",
        hint_text="Enter your prompt here",
        multiline=True,
        on_change=on_change
    )

    oa_sb = ft.ElevatedButton(
        text="Submit",
        style=sub_but_style,
        on_click=sub_prompt
    )
    oa_rt = ft.TextField(
        width=page.width-100,
        label="Generated Script",
        multiline=True,
        read_only=True
    )
    oa_col = ft.Column(
        width=page.width,
        controls=[
            oa_title, oa_sub
        ],
        spacing=10,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        alignment=ft.MainAxisAlignment.CENTER
    )
    oa_sp = ft.Text("", size=30)
    oa_col2 = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        width=page.width-100,
        controls=[
            oa_tb, oa_sb, oa_rt
        ],
        spacing=10,
        horizontal_alignment=ft.CrossAxisAlignment.START,
        alignment=ft.MainAxisAlignment.CENTER
    )
    oa_col3 = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        width=page.width,
        controls=[oa_col, oa_sp, oa_col2],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        alignment=ft.MainAxisAlignment.CENTER
    )
    gp=ft.LinearGradient(
        begin=ft.alignment.top_center,
        end=ft.alignment.bottom_center,
        rotation=-45,
        colors=[ft.colors.PURPLE, ft.colors.LIGHT_BLUE_700],
    )
    oa_page_cont=ft.Column(
            scroll=ft.ScrollMode.ALWAYS,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=20,
            controls=[
                oa_sp,
                oa_col3
            ]
        )
    oa_page = ft.Container(
        width=page.width,
        height=page.height,
        gradient=gp,
        content = oa_page_cont
        
    )
    return oa_page
    

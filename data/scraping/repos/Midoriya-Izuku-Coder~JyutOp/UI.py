import flet as ft
import Views
import openai
openai.api_key = "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"
messages = [{"role": "system", "content": "你是一個大型語言模型，具有寫很好的粵劇劇本歌詞的能力，可以寫很好的粵劇歌詞，以歌詞為對白，使用文言文的風格，要有音韻之美。"}]
# messages.append({"role": "user", "content": "按以下要求寫一個粵劇劇本歌詞並爲其命名。{}。".format(prompt)})
# import openai 
# openai.api_key = "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"
#deep purple = 504278
#light purple = 807095
#grey = F9F9F8
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
curr_page = 0
def main(page: ft.Page):
    global curr_page, mb_def_style, mb_mod_style
    page.title = "JyutOp UI"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.MainAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.ALWAYS
    page.padding = 0
    page.bgcolor = "#504278"
    top_buttons = 0
    def page_ho(e):
        page.go('/')
        pass
    def page_oe(e):
        page.go('/oe')
        pass
    def page_oa(e):
        page.go('/oa')
        pass
    def page_au(e):
        page.go('/au')
        pass
    def route_change(route):
        page.views.clear()
        page.views.append(
            Views.views_handler(page)[page.route]
        )
    page.on_route_change=route_change
    page.go('/')
ft.app(target=main)

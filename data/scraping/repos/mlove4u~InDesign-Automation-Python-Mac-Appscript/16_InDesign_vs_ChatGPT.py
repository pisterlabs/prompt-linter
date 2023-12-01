"""
Select a text frame in InDesign, call OpenAI's API, and translate it into the desired language.
"""
import wx
from appscript import *
indd = app("Adobe InDesign 2023")

import openai
openai.api_key = "**********"


def openai_translate(lan, contents):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"下記の文章を{lan}に翻訳してください"},
            {"role": "user", "content": contents},
        ],
    )
    return response.choices[0]["message"]["content"]


class MyFrame (wx.Frame):

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=-1, title="ChatInDesign",
                          size=wx.Size(300, 400))

        bSizer1 = wx.BoxSizer(wx.VERTICAL)
        #
        languageChoices = ["英語", "中国語", "韓国語"]
        self.language = wx.RadioBox(self, -1, u"Translate to", wx.DefaultPosition,
                                    wx.DefaultSize, languageChoices, 1, wx.RA_SPECIFY_ROWS)
        self.language.SetSelection(0)

        bSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.m_button1 = wx.Button(self, -1, "翻訳")
        bSizer2.Add(self.m_button1, 0, wx.ALL, 5)
        self.m_button2 = wx.Button(self, -1, "InDesignに戻す")
        bSizer2.Add(self.m_button2, 0, wx.ALL, 5)

        self.tc = wx.TextCtrl(self, -1, style=wx.TE_MULTILINE)
        #
        bSizer1.Add(self.language, 0, wx.ALL, 5)
        bSizer1.Add(bSizer2, 0, wx.EXPAND, 5)
        bSizer1.Add(self.tc, 1, wx.ALL | wx.EXPAND, 5)

        self.SetSizer(bSizer1)
        self.Layout()

        self.Centre(wx.BOTH)

        self.m_button1.Bind(wx.EVT_BUTTON, self.translate)
        self.m_button2.Bind(wx.EVT_BUTTON, self.return_to_indesign)

    def translate(self, event):
        if len(indd.selection()) == 0:
            wx.MessageBox("Please select a text_frame", "")
            return
        sel = indd.selection()[0]
        if sel.class_() != k.text_frame:
            wx.MessageBox("Please select a text_frame", "")
            return
        lan = self.language.GetStringSelection()
        contents = sel.contents()  # contents of InDesign text_frame
        r = openai_translate(lan, contents)  # translated text
        self.tc.SetValue(r)

    def return_to_indesign(self, event):
        indd.selection()[0].contents.set(self.tc.GetValue())


if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame(None)
    frame.Show()
    app.MainLoop()

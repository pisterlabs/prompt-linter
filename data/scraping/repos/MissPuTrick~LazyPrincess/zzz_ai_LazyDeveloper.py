from utils import temp
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from pyrogram import Client, filters
from info import *
import openai
openai.api_key = OPENAI_API

@Client.on_message(filters.private & filters.text)
async def lazy_answer(client, message):
    if AI == True: 
        user_id = message.from_user.id
        if user_id:
            try:
                lazy_users_message = message.text
                user_id = message.from_user.id
                response = openai.Completion.create(
                    model = "text-davinci-003",
                    prompt = lazy_users_message,
                    temperature = 0.5, 
                    max_tokens = 1000,
                    top_p=1,
                    frequency_penalty=0.1,
                    presence_penalty = 0.0,
                )
                btn=[
                        [InlineKeyboardButton(text=f"⇱🤷‍♀️ Mengambil tindakan 🗃️⇲", url=f'https://t.me/{temp.U_NAME}')],
                        [InlineKeyboardButton(text=f"🗑 Hapus log ❌", callback_data=f'close_data')],
                    ]
                reply_markup=InlineKeyboardMarkup(btn)
                footer_credit = "🦋<a href='https://telegram.me/TMID_CSbot'>• MASALAH LAPORAN •</a>══<a href='https://telegram.me/TMID_CSbot'>• HUBUNGI OWNER •</a>🦋"
                lazy_response = response.choices[0].text 
                await client.send_message(LAZY_AI_LOGS, text=f"Seorang pengguna bernama **{message.from_user.mention}** dengan id pengguna - `{user_id}`. Menanyakan pertanyaan ini kepada saya...\n\n══❚█══Q   U   E   R   Y══█❚══\n\n\n[Q྿.]**{lazy_users_message}**\n\n👇Inilah yang saya tanggapi:\n:-`{lazy_response}`\n\n\n❚═IDENTITAS PENGGUNA═❚═• `{user_id}` \n❚═NAMA BELAKANG═❚═• `{message.from_user.mention}` \n\n🗃️" , reply_markup = reply_markup )
                await message.reply(f"{lazy_response}\n\n\n{footer_credit}")
            except Exception as error:
                print(error)
    else:
        return
    
























    







        # i am NOT INTRESTED IN BEING YOUR SECOND FATHER... SO DON'T REMOVE MY CREDIT...
        #                               @LazyDeveloperr











        #                                        ...PRESENTING...
        #                          A R T I F i C I A L  -  I N T E L i G E N C E
        #                                      .in  Auto-Filter-Bot.
        #                                         @LazyDeveloperr

                                      










        # this line is for copy-pasters...
        # ...while you are removing my credit and calling yourself a developerr... 
        # _____ just imagine, At that time i am fucking your mom and sis at same time, harder & too harder...





































#
#         i am NOT INTRESTED IN BEING YOUR SECOND FATHER... SO DON'T REMOVE MY CREDIT...
#                                       @LazyDeveloperr
# 
# 
# 
# 
#
#
#
#
#
#
#
#                                                ...PRESENTING...
#                                  A R T I F i C I A L  -  I N T E L i G E N C E
#                                              .in  Auto-Filter-Bot.
#                                                 @LazyDeveloperr
#
#                                       
#
#
#
#
#
#
#
#
#
#
#         this line is for copy-pasters...
#         ...while you are removing my credit and calling yourself a developerr... 
#         _____ just imagine, At that time i am fucking your mom and sis at same time, harder & too harder...
#

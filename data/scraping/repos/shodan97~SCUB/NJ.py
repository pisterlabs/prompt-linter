import re
import time
import openai
import random
import stripe
import aiohttp
from telethon import TelegramClient, events, functions, Button
from alive_progress import alive_bar
from telethon.tl.functions.account import UpdateProfileRequest
from telethon.tl.functions.photos import UploadProfilePhotoRequest
from telethon.tl.types import User, MessageEntityCustomEmoji
from telethon import types

from telethon.tl.functions.channels import InviteToChannelRequest
from telethon.tl.types import InputPeerEmpty, InputPeerChannel, InputPeerUser

from faker import Faker

import asyncio

from art import *
from data.config import *

openai.api_key = "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"
scrapcc_chat_id = -1002074810067


client = TelegramClient('Sc', api_id, api_hash)
client.parse_mode = 'html'

cards_scraped = []

pattern = r"(?:^[0-9]{6,19}$)|([0-9]{13,19})[\s|\S]{0,3}?([0-9]{1,2})[\s|\S]{0,3}?([0-9]{2,4})[\s|\S]{0,3}?([0-9]{3,4})"

base_nonvbv = ['440507', '488893', '440066', '411568', '456835', '546540', '531106', '549636', '517805', '402074', '601120', '414711', '414720', '424631', '438857', '426428', '540168', '482860', '485620', '426685', '426684', '400022', '430023', '551149', '442742', '483316', '449465', '434256', '456367', '542418', '474488', '446542', '488853', '519955', '611291', '453903', '414709', '546616', '601100', '411773', '406879', '412443', '413580', '415285', '422510', '526298', '532296', '534860', '422991', '427110', '449185', '444784', '402298', '401854', '403492', '494053', '483899', '476091', '456487', '456485', '456482', '456481', '456472', '456469', '456409', '455701', '453030', '451829', '439240', '435216', '426556', '426033', '412978', '409774', '407220', '494398', '494133', '493172', '493171', '493164', '493161', '493160', '493158', '493135', '491871', '491584', '491572', '491512', '491502', '491405', '491399', '493404', '491688', '477950', '434495', '431887', '431771', '428988', '423128', '420584', '413463', '413443', '413397', '404251', '403328', '403084', '494331', '497849', '497848', '497847', '497845', '497844', '497843', '497841', '497840', '497839', '497838', '497835', '497833', '497831', '497830', '497825', '497823', '497822', '497821', '497820', '497819', '489765', '486791', '485178', '483347', '482500', '482070', '482052', '482028', '482024', '481996', '480684', '480659', '480392', '478864', '475800', '473549', '473489', '471927', '471925', '471923', '559184', '490249', '459077', '515225', '442317', '426154', '536431', '434662', '517954', '493889', '420885', '438387', '220070', '451014', '523641', '451223', '516949', '423120', '400446', '402306', '559958', '410859', '551106', '549123', '537040', '541590', '523914', '516948', '512413', '497538', '473872', '477918', '468597', '451407', '449105', '435583', '417664', '417663', '417662', '417661', '417660', '479126', '410894', '427138', '480011', '439102', '498235', '478192', '476164', '432845', '432630', '456678', '406095', '462120', '465614', '435237', '456432', '456430', '460198', '456444', '443446', '455702', '414790', '450004', '424698', '498416', '463576', '463572', '478455', '463506', '422005', '463575', '430858', '471630', '515598', '516361', '550200', '521894', '512576', '516320', '516329', '544637', '528013', '543568', '546827', '555005', '540909', '531355', '529149', '514616', '556963', '564545', '576135', '551896', '552234', '546068', '543805', '455620', '415974', '523236', '455600', '456874', '490638', '420208', '413585', '546812', '415423', '428258', '427742', '465596', '419403', '462730', '460312', '520309', '542555', '546058', '455554', '464440', '436501', '436542', '414260', '413777', '447963', '448400', '425727', '440669', '447965', '448448', '485738', '447964', '445984', '491002', '542061', '542064', '546297', '546298', '547872', '516029', '486483', '455598', '454434', '454818', '494052', '456448', '443431', '443451', '468524', '409349', '493414', '443420', '552638', '456442', '450605', '443458', '421699', '456443', '521729', '443462', '443440', '443438', '421307', '541576', '421663', '523225', '525615', '421312', '424911', '424201', '426163', '456428', '471563', '448275', '448210', '424604', '478880', '479849', '448666', '448670', '442813', '441103', '491991', '488890', '480239', '480174', '487093', '479804', '479853', '412299', '474398', '493174', '482862', '428995', '473690', '403995', '400806', '402203', '403461', '403497', '404227', '470712', '480213', '480260', '471562', '427533', '478821', '414736', '479162', '482880', '431303', '426627', '455592', '482857', '447619', '451046', '431247', '469594', '518840', '408586', '408089', '301020', '543448', '546689', '451477', '552090', '554860', '411845', '544859', '552000', '426588', '400261', '498406', '490172', '422007', '552072', '552640', '552289', '400192', '434812', '516026', '416027', '416028', '416029', '426389', '434994', '438284', '440769', '440797', '490824', '415501', '417624', '520639', '528910', '540609', '548327', '428259', '450760', '589973', '529925', '458109', '411824', '451607', '544612', '518127', '549191', '492268', '492271', '492273', '492275', '492277', '492280', '492282', '492284', '492286', '493799', '498000', '498011', '498020', '498021', '498028', '498034', '498037', '498039', '498041', '498043', '498045', '498047', '498049', '498051', '498053', '498055', '498057', '498059', '498071', '498074', '498077', '498079', '498080', '498607', '441105', '511958', '426354', '549003', '548696', '552554', '526624', '526615', '402360', '410666', '547347', '450831', '530127', '414258', '414259', '479741', '498823', '492181', '542502', '454313', '465838', '465859', '465923', '446261', '543458', '446280', '492182', '446274', '465910', '545140', '421182', '542011', '446278', '455572', '450906', '552188', '543460', '454369', '529930', '454638', '492083', '492207', '455272', '557843', '550534', '548041', '550566', '552073', '550142', '377064', '379102', '379109', '377311', '374692', '374285', '375395', '497599', '497092', '497084', '497037', '497023', '497030', '497038', '497568', '405784', '497230', '497519', '497539', '497566', '497059', '497576', '497547', '497128', '497587', '497052', '456480', '454605', '425135', '544434', '545240', '545241', '545242', '545243', '545270', '549198', '549199', '405221', '443460', '516323', '517969', '521893', '535316', '440260', '453980', '491611', '546258', '528070', '541332', '546002', '549154', '403315', '403346', '406357', '474361', '514876', '548213', '548596', '543499', '543662', '545307', '547015', '547046', '547066', '547146', '547484', '554900', '557905', '557908', '455644', '479273', '490845', '528055', '417325', '423896', '450653', '454891', '492019', '518706', '400188', '401849', '401850', '402057', '402058', '402269', '402270', '402271', '403586', '403632', '403649', '403650', '406020', '408541', '412912', '414909', '425970', '427208', '427340', '427718', '428972', '434496', '440657', '440768', '440774', '460313', '460314', '460315', '477574', '479707', '490158', '490804', '491075', '491224', '491225', '491782', '491820', '491821', '493476', '493713', '494116', '494342', '494350', '523253', '526838', '553133', '553390', '553403', '424607', '541330', '542223', '547503', '552885', '436774', '456492', '498873', '403015', '403016', '492578', '492589', '405091', '421199', '422165', '424339', '426387', '426388', '431208', '453900', '457002', '458117', '458190', '491889', '520952', '529936', '542542', '543077', '543560', '544313', '554268', '554477', '554501', '554758', '431431', '434960', '510021', '513659', '540490', '550011', '402944', '458090', '429672', '480327', '421701', '425908', '408104', '415747', '439707', '425489', '432423', '486236', '446053', '479030', '542432', '454337', '451401', '552672', '441281', '468018', '419310', '453825', '428434', '545534', '482854', '447091', '433438', '520416', '455015', '403444', '438567', '554302', '498453', '462161', '524886', '546626', '528725', '457949', '420767', '456323', '520953', '458010', '458003', '547182', '497598', '513379', '513283', '415056', '472926', '456268', '456242', '513263', '412039', '414718', '443057', '462406', '449063', '442299', '448027', '413789', '430921', '482870', '426998', '487390', '423903', '443658', '467541', '430550', '472362', '443220', '432388', '448858', '425379', '432372', '488860', '436678', '432371', '486830', '426401', '432671', '432383', '451210', '467546', '476165', '486827', '404562', '447094', '440893', '415384', '482853', '447972', '491986', '405813', '447010', '446024', '412185', '466309', '482863', '449053', '449221', '415491', '406058', '480613', '449052', '467616', '449449', '427509', '485403', '422968', '445101', '447261', '449477', '489262', '482864', '407204', '487309', '448869', '432739', '409311', '446540', '412061', '443239', '426429', '485671', '401171', '449880', '431307', '400079', '401436', '405302', '409782', '410229', '410953', '411352', '411713', '429441', '444468', '445452', '448232', '451769', '453979', '454742', '459465', '465858', '473702', '475714', '479091', '479769', '484799', '515676', '522860', '527269', '530900', '532186', '539132', '547087', '551606', '553608', '677671', '409015', '425723', '435659', '450608', '450625', '450663', '453243', '454108', '454617', '454618', '455218', '455221', '455262', '456140']




		

@client.on(events.NewMessage(pattern=r'[.!/].*fake'))
async def fake_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		message_text = event.message.text[6:]
		
		fake = Faker('en_US')
		await client.edit_message(event.message, f'<b>🙋🏼‍♂️ Name: <code>{fake.name()}</code>\n⛺️ Address: <code>{fake.street_address()}</code>\n🌏 Country: <code>{fake.country()}\n🏙 City: <code>{fake.city()}</code>\n☁️ State: <code>{fake.state()}</code>\n🗃 Post code: <code>{fake.postcode()}</code></b>')


@client.on(events.NewMessage(pattern=r'[.!/].*type'))
async def type_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		message_text = event.message.text[6:]

		text = ''

		for char in message_text:
			try:
				text += char
				await client.edit_message(event.message, f'<b>{text[:-1]}░</b>')
				time.sleep(.05)
				await client.edit_message(event.message, f'<b>{text}</b>')
				time.sleep(.05)

			except FloodWait as e:
				time.sleep(e.x)


@client.on(events.NewMessage(pattern=r'[.!/].*setname'))
async def setname_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		message_text = event.message.text
		name = event.message.text[9:]

		await client(UpdateProfileRequest(first_name=name))

		await client.edit_message(event.message, f"<b>✔️ Name update - <code>{name}</code></b>")


@client.on(events.NewMessage(pattern=r'[.!/].*spam'))
async def test_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		message_text = event.message.text
		count = message_text.split()[1]
		text = event.message.text[7+len(count):]

		x = 0

		while x < int(count):
			try:
				await client.send_message(chat, f'<b>{text}</b>')
				x += 1
			
			except FloodWait as e:
				time.sleep(e.x)


@client.on(events.NewMessage(pattern=r'[.!/].*svalid'))
async def scrap_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		message_text = event.message.text
		try:
			if id_chat := message_text.split()[1]:
				channel = await client.get_entity(int(id_chat))
				try:
					name = channel.title
				except:
					name = channel.first_name
				
				await client.edit_message(event.message, f"<b>♻️ I'm starting scrap {name} [Only Valid]</b>")

				cards = []

				messages = await client.get_messages(channel, limit = None)

				for message in messages:
					if message != None:
						card = re.findall(pattern, str(message.message))
						for math in card:
							number, month, year, cvv = math
							if len(year) == 4: year = year[2:]
							if len(month) == 1: month = '0'+month
							full_card = '{}|{}|{}|{}'.format(number, month, year, cvv)
							if full_card not in cards:
								if '✅' in message.message:
									cards.append(full_card)
									print(full_card)

				if len(cards) != 0:
					result = '\n'.join(cards)
					with open(f'scraped[ninja].txt', 'w') as f:
						f.write(result)
						f.close()

					await client.send_file(event.chat_id, file=open('scraped[ninja].txt', 'rb'))
					await client.edit_message(event.message, f'<b>✔️ Scraping cards from {name} finished.\n\n💳 Total cards - {len(cards)}</b>')
				else:
					await client.edit_message(event.message, f'<b>⚠️ No cards found</b>')
		except Exception as e:
			print(e)
			await client.edit_message(event.message, f'<b>❕ Incorrect chat ID</b>')

@client.on(events.NewMessage(pattern=r'[.!/].*scrap'))
async def scrap_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		message_text = event.message.text
		try:
			if id_chat := message_text.split()[1]:
				channel = await client.get_entity(int(id_chat))
				try:
					name = channel.title
				except:
					name = channel.first_name
				
				await client.edit_message(event.message, f"<b>♻️ I'm starting scrap {name}</b>")
				
				cards = []

				messages = await client.get_messages(channel, limit = None)

				for message in messages:
					if message != None:
						card = re.findall(pattern, str(message.message))
						for math in card:
							number, month, year, cvv = math
							if len(year) == 4: year = year[2:]
							if len(month) == 1: month = '0'+month
							full_card = '{}|{}|{}|{}'.format(number, month, year, cvv)
							if full_card not in cards:
								cards.append(full_card)


				if len(cards) != 0:
					result = '\n'.join(cards)
					with open(f'scraped[ninja].txt', 'w') as f:
						f.write(result)
						f.close()

					await client.send_file(event.chat_id, file=open('scraped[ninja].txt', 'rb'))
					await client.edit_message(event.message, f'<b>✔️ Scraping cards from {name} finished.\n\n💳 Total cards - {len(cards)}</b>')
				else:
					await client.edit_message(event.message, f'<b>⚠️ No cards found</b>')
		except Exception as e:
			print(e)
			await client.edit_message(event.message, f'<b>❕ Incorrect chat ID</b>')


@client.on(events.NewMessage(pattern=r'[.!/].*add'))
async def add_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		message_text = event.message.text
		try:
			if id_chat := message_text.split()[1]:
				channel = await client.get_entity(int(id_chat))
				await client.edit_message(event.message, f"<b>♻️ I'm starting add scrap {channel.title}</b>")
				
				cards = []

				messages = await client.get_messages(channel, limit = None)

				for message in messages:
					if message != None:
						card = re.findall(pattern, str(message.message))
						for math in card:
							number, month, year, cvv = math
							if len(year) == 4: year = year[2:]
							if len(month) == 1: month = '0'+month
							full_card = '{}|{}|{}|{}'.format(number, month, year, cvv)
							if full_card not in cards_scraped:
								cards_scraped.append(full_card)
								cards.append(full_card)

				if len(cards) != 0:
					await client.edit_message(event.message, f'<b>✔️ Adding cards from {channel.title} finished.\n\n💳 Cards added/base - {len(cards)} / {len(cards_scraped)}</b>')
				else:
					await client.edit_message(event.message, f'<b>⚠️ No cards found or cards are already in the base</b>')
		except Exception as e:
			print(e)
			await client.edit_message(event.message, f'<b>❕ Incorrect chat ID</b>')

@client.on(events.NewMessage(pattern=r'[.!/].*export'))
async def export_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		await client.edit_message(event.message, f"<b>♻️ Unloading cards...</b>")

		if len(cards_scraped) != 0:
			result = '\n'.join(cards_scraped)
			with open(f'scraped[ninja].txt', 'w') as f:
				f.write(result)
				f.close()

			await client.send_file(event.chat_id, file=open('scraped[ninja].txt', 'rb'))
			await client.edit_message(event.message, f"<b>✔️ Cards export completed\n\n💳 Cards exported - {len(cards_scraped)}</b>")

@client.on(events.NewMessage(pattern=r'[.!/].*count'))
async def count_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		await client.edit_message(event.message, f"<b>✔️ Cards in base - {len(cards_scraped)}</b>")

@client.on(events.NewMessage(pattern=r'[.!/].*clear'))
async def clear_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		await client.edit_message(event.message, f"<b>✔️ Cards cleared - {len(cards_scraped)}</b>")
		cards_scraped.clear()

@client.on(events.NewMessage(pattern=r'[.!/].*chatid'))
async def chatid_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		await client.edit_message(event.message, f"<b>✔️ Chat id - <code>{event.chat_id}</code></b>")

@client.on(events.NewMessage(pattern=r'[.!/].*antip'))
async def chatid_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		reply_message = await event.get_reply_message()
		if reply_message:

			data = []

			public = ''
			private = ''

			card = re.findall(pattern, str(reply_message.message))
			for math in card:
				number, month, year, cvv = math
				if len(year) == 4: year = year[2:]
				if len(month) == 1: month = '0'+month

				data.append(number)
			
			async with aiohttp.ClientSession() as session:
				async with session.post(f'https://api.antipublic.cc/cards', json=data) as antip:
					result = await antip.json()
					for math in card:
						number, month, year, cvv = math
						full_card = '{}|{}|{}|{}'.format(number, month, year, cvv)
						if len(year) == 4: year = year[2:]
						if len(month) == 1: month = '0'+month
						
						if number in result['public']:
							public += f'{full_card}\n'
						else:
							private += f'{full_card}\n'


						
					
					percentage = result['private_percentage']
					count_private = len(result['private'])
					count_public = len(result['public'])

			if public == '':
				await client.edit_message(event.message, f"<b>📗 Private [{count_private}]:\n<code>{private}</code>\n✔️ Private percentage - {percentage:.2f}%</b>")
			elif private == '':
				await client.edit_message(event.message, f"<b>📙 Public [{count_public}]:\n<code>{public}</code>\n✔️ Private percentage - {percentage:.2f}%</b>")
			else:
				await client.edit_message(event.message, f"<b>📗 Private [{count_private}]:\n<code>{private}</code>\n📙 Public [{count_public}]:\n<code>{public}</code>\n✔️ Private percentage - {percentage:.2f}%</b>")

@client.on(events.NewMessage(pattern=r'[.!/].*key'))
async def key_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		message_text = event.message.text
		key = message_text.split()[1]
		hidden_key = f'{key[:20]}********************************************************{key[-1]}'

		stripe.api_key = key

		try:
			x = stripe.PaymentMethod.create(type="card", card={ "number": "5581585437580232", "exp_month": '06', "exp_year": '26', "cvc": "123"})
			await client.edit_message(event.message, f'<b>✅ Live SK\n\n📂 Key: <code>{hidden_key}</code></b>')
		except Exception as e:
			print(e)
			if 'Expired API Key provided' in str(e):
				await client.edit_message(event.message, f'<b>❌ Dead SK\n\n📂 Key: <code>{key}</code>\nMessage - Expired API Key provided</b>')
			elif 'Your account cannot currently make live charges' in str(e):
				await client.edit_message(event.message, f'<b>❌ Dead SK\n\n📂 Key: <code>{key}</code>\nMessage - Your account cannot currently make live charges</b>')
			else:
				msg = str(e).split(':')[1].strip().split('.')[0]
				await client.edit_message(event.message, f'<b>❌ Dead SK\n\n📂 Key: <code>{key}</code>\nMessage - Invalid API Key provided</b>')

@client.on(events.NewMessage(pattern=r'[.!/].*bin'))
async def bin_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		message_text = event.message.text
		bin = ''.join(re.findall(r'([1-9]{1}[0-9]{5})', message_text))

		if len(bin) < 6:
			await client.edit_message(event.message, f'<b>❕ Invalid bin/b>')
		else:
			async with aiohttp.ClientSession() as session:
				async with session.get(f'https://bins.antipublic.cc/bins/{bin}') as bin_check:
					data = await bin_check.json()
					if 'detail' in data:
						await client.edit_message(event.message, f'<b>❕ Invalid bin</b>')
					else:
						bin_number = data['bin']
						brand = data['brand']
						type = data['type']
						level = data['level']
						bank = data['bank']
						country = data['country_name']
						country_flag = data['country_flag']

						text = f'<b>📒 BIN: {bin_number}</b>\n'

						if brand != None:
							text += f'<b>♻️ TYPE: {brand}</b>'
						else:pass
						if type != None:
							text += f'<b>, {type}</b>\n'
						else:
							text += f'\n'
						if level != None:
							text += f'<b>🔥 LEVEL: {level}</b>\n'
						else:pass
						if bank != None:
							text += f'<b>💵 BANK: {bank}</b>\n'
						else:pass
						if country != None:
							text += f'<b>🌎 COUNTRY: {country} {country_flag}</b>\n'
						else:pass
						if bin_number in base_nonvbv:
							text += f'\n<b>NON/AUTO VBV: ✔️</b>\n'
						else:pass

						await client.edit_message(event.message, text)

@client.on(events.NewMessage())
async def card_search_message(event):
	try:
		message = str(event.message.message)

		card = re.findall(pattern, message)
		for math in card:
			number, month, year, cvv = math
			if len(year) == 4: year = year[2:]
			if len(month) == 1: month = '0'+month
			full_card = '{}|{}|{}|{}'.format(number, month, year, cvv)
			if full_card not in cards_scraped:
				cards_scraped.append(full_card)
				async with aiohttp.ClientSession() as session:
					async with session.get(f'https://bins.antipublic.cc/bins/{number}') as bin_check:
						data = await bin_check.json()
						if 'detail' in data:
							return
						else:
							bin_number = data['bin']
							brand = data['brand']
							type = data['type']
							level = data['level']
							bank = data['bank']
							country = data['country_name']
							country_flag = data['country_flag']

							await client.send_message(scrapcc_chat_id, f'<b>• Card\n    ↳ <code>{full_card}</code>\n• Bin Information\n    ↳ Type - <code>{brand}, {type}</code>\n    ↳ Level - <code>{level}</code>\n    ↳ Bank - <code>{bank}</code>\n    ↳ Country - <code>{country} {country_flag}</code></b>')

		if re.search(r'CryptoBot', message):
			regex = re.search(r'CQ\S+', message)
			check = regex.group(0)
			if len(check) > 12:
				x = len(check) - 12
				check = check[:-x]
			await client.send_message('CryptoBot', '/start ' + check)
		elif re.search(r'tonRocketBot', message):
			if re.search(r't_', message):
				regex = re.search(r't_\S+', message)
				check = regex.group(0)
				if len(check) > 17:
					x = len(check) - 17
					check = check[:-x]
				await client.send_message('tonRocketBot', '/start ' + check)
			elif re.search(r'mc_', message):
				regex = re.search(r'mc_\S+', message)
				check = regex.group(0)
				if len(check) > 18:
					x = len(check) - 18
					check = check[:-x]
				await client.send_message('tonRocketBot', '/start ' + check)
			elif re.search(r'mci_', message):
				regex = re.search(r'mci_\S+', message)
				check = regex.group(0)
				if len(check) > 19:
					x = len(check) - 19
					check = check[:-x]
				await client.send_message('tonRocketBot', '/start ' + check)
		elif re.search(r'wallet', message):
			if re.search(r'C-', message):
				regex = re.search(r'C-\S+', message)
				check = regex.group(0)
				if len(check) > 12:
					x = len(check) - 12
					check = check[:-x]
				await client.send_message('wallet', '/start ' + check)
		elif re.search(r'RandomTGbot', message):
			id = message.replace('http://t.me/RandomTGbot?start=', '')
			await client.send_message('RandomTGbot', '/start ' + id)



		if event.message.buttons:
			message = event.message.buttons[0][0].url
			if re.search(r'CryptoBot', message):
				regex = re.search(r'CQ\S+', message)
				check = regex.group(0)
				if len(check) > 12:
					x = len(check) - 12
					check = check[:-x]
				await client.send_message('CryptoBot', '/start ' + check)
			elif re.search(r'tonRocketBot', message):
				if re.search(r't_', message):
					regex = re.search(r't_\S+', message)
					check = regex.group(0)
					if len(check) > 17:
						x = len(check) - 17
						check = check[:-x]
					await client.send_message('tonRocketBot', '/start ' + check)
				elif re.search(r'mc_', message):
					regex = re.search(r'mc_\S+', message)
					check = regex.group(0)
					if len(check) > 18:
						x = len(check) - 18
						check = check[:-x]
					await client.send_message('tonRocketBot', '/start ' + check)
				elif re.search(r'mci_', message):
					regex = re.search(r'mci_\S+', message)
					check = regex.group(0)
					if len(check) > 19:
						x = len(check) - 19
						check = check[:-x]
					await client.send_message('tonRocketBot', '/start ' + check)
			elif re.search(r'wallet', message):
				if re.search(r'C-', message):
					regex = re.search(r'C-\S+', message)
					check = regex.group(0)
					if len(check) > 12:
						x = len(check) - 12
						check = check[:-x]
					await client.send_message('wallet', '/start ' + check)
	except Exception as e:
		pass

@client.on(events.NewMessage(pattern=r'[.!/].*lol'))
async def lol_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id

	all_participants = await client.get_participants(chat)
	random_user = random.choice(all_participants)
	if user_id in admins:
		x = 0
		emoji = ['⚪️', '🟢']

		while x < 100:
			try:
				await client.edit_message(event.message, f'<b>{random.choice(emoji)} Поиск дауна... {x}%</b>')
				x += random.randint(1, 3)
				time.sleep(0.1)
			
			except FloodWait as e:
				time.sleep(e.x)
		try:
			await client.edit_message(event.message, f'<b>🟢 Даун найден - @{random_user.username}</b>')
		except:
			await client.edit_message(event.message, f'<b>🟢 Даун найден - {random_user.first_name}</b>')

@client.on(events.NewMessage(pattern=r'[.!/].*gpt'))
async def gbt_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		message_text = event.message.text[5:]
		await client.edit_message(event.message, f'<b>♻️ Please wait...</b>')
		response = openai.ChatCompletion.create(
			model = "gpt-3.5-turbo",
			messages=[
			{"role": "user", "content": ""+message_text+""}
			]
		)

		await client.edit_message(event.message, f'<b>✔️ ChatGPT:\n\n<code>{response["choices"][0]["message"]["content"]}</code>\n\n❔ Your question to ChatGPT: <code>{message_text}</code></b>')


@client.on(events.NewMessage(pattern=r'[.!/].*gpt_img'))
async def img_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		message_text = event.message.text[5:]
		prompt = message_text
		if prompt.strip() == '':
			await client.edit_message(event.message, '<b>❕ Send what photo you want to take</b>')
		else:
			await client.edit_message(event.message, f'<b>♻️ Please wait...</b>')
			response = openai.Image.create(
				prompt=prompt,
				n=1,
				size="1024x1024"
			)
			image_url = response['data'][0]['url']
			
			await client.send_file(chat, file=image_url, force_document=False)
			await client.edit_message(event.message, f'<b>✔️ Generated success - <code>{message_text}</code></b>')

@client.on(events.NewMessage(pattern=r'[.!/].*mts'))
async def scrap_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		await client.edit_message(event.message, f"<b>♻️ Search...</b>")
		numbers = re.findall(r'[0-9]{10}', event.message.text)

		cookies = {
			'.AspNetCore.RegionId': 'MOW',
			'.AspNetCore.Antiforgery.q1sNu47QVQs': 'CfDJ8N4DLLnIWtRMt-Q_2fsioQ_FZ7SXB98pPbSHIF90TaT_S6odo1tTNxrve7zSxojvXQZ6SqC1htkTbzcknoDnOUBIq2hvMl0FjUpWTcG9AcglnGKHf_iQc3HuVrHVIRHxMs51_voHR04YYXM1LQJrYp4',
			'StickyID': '!MXm137itRJWrOvkpFL1Kx71Z9S8d2iCTOhPHggFlwbJl3psk0Yt0Uvqn7gFFhb8otJmJfITOJ1JQQA==',
			'TS016bb39a': '012019f3d4ccf1fecfc599ae62a3a4cb508b7b26edaafcd8ce13ad810d81023498fea68ca4c9efe715e3f433031d4b393c0e1a6b0ea036a615699c68a3d7195a3f27106a33239db55e12bafd2e6d3f31e864e91cb6650db8e404f7d8b63bbedec54930c862fe4482f2fc7298706e0115dae56f9635',
			'.AspNetCore.LocalStorage': 'eyJmZWVkYmFjay1zZXNzaW9uIjoiMzQ1MmRiYWQtOTgxYy0yYzc1LWUzZjEtZTAyYjYzZTM5OTAzIn0=',
			'.AspNetCore.Session': 'CfDJ8N4DLLnIWtRMt%2BQ%2F2fsioQ%2FECXNMiV9N4X%2BnEKeHlkjNVfw%2B68ptqeydef9cb2AblvqQknPqxIfkOZCuvQR2lgO7fIb18%2FxQnqQha4%2BAFrfsmHG2Iuj3ESdpNwFWHlEWo44NVuWm1y0xHisVE2m41B2X9U%2Bt3Ryirbw9j8fs7pc%2F',
			'__zzatmts-w-payment': 'MDA0dC0cTApcfEJcdGswPi17CT4VHThHKHIzd2VbQSFSHUdfUUASVXtbFhV8cldMORFhcj5vdF1vbiNlehUgQ1U/dRdZRkE2XBpLdWUJCzowJS0xViR8SylEW1R7KBsTfHAoWA8NVy8QLj9hTixcIggWEU0hF0ZaFXtDPGMMcRVNfX0mNGd/ImUrOS5sDQtyuA==',
			'__zzatmts-w-payment': 'MDA0dC0cTApcfEJcdGswPi17CT4VHThHKHIzd2VbQSFSHUdfUUASVXtbFhV8cldMORFhcj5vdF1vbiNlehUgQ1U/dRdZRkE2XBpLdWUJCzowJS0xViR8SylEW1R7KBsTfHAoWA8NVy8QLj9hTixcIggWEU0hF0ZaFXtDPGMMcRVNfX0mNGd/ImUrOS5sDQtyuA==',
			'cfidsmts-w-payment': '+yUq8qHPEvcKaXdvYZTH3PA6P+Ytw19IV7GzpXDY+jYoI/0rs+xaE8vePa3cyEW19OhgCf+Khkuysiioy4iRP95FKVzNklgZNr5WR6TucAdFiH+1kCsuanph909+/aVKobbWKbD5HYIcpy1lr8EMyTqtE0m8fCKWR3gk',
			'cfidsmts-w-payment': '+yUq8qHPEvcKaXdvYZTH3PA6P+Ytw19IV7GzpXDY+jYoI/0rs+xaE8vePa3cyEW19OhgCf+Khkuysiioy4iRP95FKVzNklgZNr5WR6TucAdFiH+1kCsuanph909+/aVKobbWKbD5HYIcpy1lr8EMyTqtE0m8fCKWR3gk',
			'gsscmts-w-payment': '2ij7N7XD4FZAI4e+Zfp9/Kgp03kIdEYqP6pflictDpb6WsluGzQ05RbT2/rDkSBGHmem/Xa7GMIXZAp7KiYODpvH/NXadzuRuQ2E1yY7/DaBfUix9EvUAm/G7U6h0JbTIV2Zxiw+Kk4zSnQ/MvWMi63JryURwJsacJbnJhSIRGXDAwGx2gmJ/EHp3H1pktZxTy3c1wX5Y/J8hVhlOfv+2x0V6dGnJfGh9OhX6SYYaqZtjslO2ZbgbH5xYXOsfBV7u69HcZ3Jn7PlWAJi92ln',
			'gsscmts-w-payment': '2ij7N7XD4FZAI4e+Zfp9/Kgp03kIdEYqP6pflictDpb6WsluGzQ05RbT2/rDkSBGHmem/Xa7GMIXZAp7KiYODpvH/NXadzuRuQ2E1yY7/DaBfUix9EvUAm/G7U6h0JbTIV2Zxiw+Kk4zSnQ/MvWMi63JryURwJsacJbnJhSIRGXDAwGx2gmJ/EHp3H1pktZxTy3c1wX5Y/J8hVhlOfv+2x0V6dGnJfGh9OhX6SYYaqZtjslO2ZbgbH5xYXOsfBV7u69HcZ3Jn7PlWAJi92ln',
			'fgsscmts-w-payment': '63wV3e27e7273049c4e0f11f60867c3b27efebd8',
			'fgsscmts-w-payment': '63wV3e27e7273049c4e0f11f60867c3b27efebd8',
		}

		headers = {
			'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36',
			'Accept': '*/*',
			'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
			'X-Requested-With': 'XMLHttpRequest',
			'Origin': 'https://payment.mts.ru',
			'Connection': 'keep-alive',
			'Referer': 'https://payment.mts.ru/pay/1583',
		}

		data = {
			'ProviderId': '11061',
			'productNameGtm': 'internet_tv_i_telefoniya_mts',
			'PaymentSumMin': '1',
			'PaymentSumMax': '60000',
			'PaymentToken': '',
			'IsZeroCommision': 'False',
			'PaymentWithCreditEnabled': 'False',
			'IsCachebackOut': 'False',
			'CheckOperatorEnabled': 'False',
			'CategoryName': 'internet_and_tv',
			'Parameters[0].Type': 'PhoneField',
			'Parameters[0].Name': 'id1',
			'Parameters[0].Val1': ''+numbers[0]+'',
			'IsApCheck': 'true',
			'Sum': '',
			'SelectedInstrumentId': 'ANONYMOUS_CARD',
			'Pan': '',
			'ExpiryMonthYear': '',
			'Cvc': '',
			'__RequestVerificationToken': 'CfDJ8N4DLLnIWtRMt-Q_2fsioQ84IefaL8knjl92Yn14GhGRWl4iEj24cGqNfOdx6JvDBuHJP7G9LjhkEFVIKfic0QKPdxuCeBnxJTmDBENwhGrXMR9aeRbdGJwChvEzoi0stw43uPhp1neTNUOyq2ENpXo',
			'FhpSessionId': 'b6def06b-e82f-457d-b66e-21dd966cd100',
			'FhpRequestId': 'ef1a29bd-3717-45a8-961d-c5e32ff7eea6',
			'Location': 'https://payment.mts.ru/pay/1583',
			'Name': 'Интернет, ТВ и Телефония МТС',
		}

		async with aiohttp.ClientSession() as session:
			async with session.post(f'https://payment.mts.ru/payment/payform/GetProviderBalance', cookies=cookies, headers=headers, data=data) as result:
				data = await result.json()
				
				if data['success'] == True:
					await client.edit_message(event.message, f"<b>📞 Phone - <code>{numbers[0]}</code>\n\n❔ Balance - {data['balance']} RUB</b>")
				else:
					await client.edit_message(event.message, f"<b>📞 Phone - <code>{numbers[0]}</code>\n\n❔ Result - Incorrect phone number</b>")

@client.on(events.NewMessage(pattern='[.!/].*spam'))
async def spam_commands(event):
    chatid = int(event.message.from_id.user_id)
    if chatid == admin:
        try:
            razdel = event.text.split()
            print(razdel)
            await event.delete()
            count = int(razdel[1])
            reply = await event.get_reply_message()
            if reply:
                if reply.media:
                    for q in range(count):
                        await client.send_file(event.to_id, reply.media)
                else:
                    for q in range(count):
                        await client.send_message(event.to_id, reply)
            else:
                msg = ' '.join(razdel[2:])
                for q in range(count):
                    await client.send_message(event.to_id, f"{msg}")
        except:
            await client.send_message(event.to_id, ".spam [Кол-во] [Сообщение]", parse_mode="HTML")

@client.on(events.NewMessage(pattern=r'[.!/].*help'))
async def info_message(event):
	chat = await event.get_chat()
	sender = await event.get_sender()
	user_id = sender.id
	if user_id in admins:
		await client.edit_message(event.message,
			'''<b>
╭──────────────────────────╮
│ ᴄᴏᴍᴀɴᴅs ɴɪɴᴊᴀ ᴜsᴇʀ ʙᴏᴛ
├──────────────────────────┤
│ <code>.scrap [id]</code> - Scrap cards from the chat
│ <code>.add [id]</code> - Add cards to the base
│ <code>.export</code> - Export cards from the base
│ <code>.count</code> - Size of cards in the base
│ <code>.clear</code> - Clear the base
├──────────────────────────┤
│ <code>.bin</code> - Checking card bin
│ <code>.key</code> - Stripe key check
│ <code>.mts</code> - Balance and check mts phone
│ <code>.antip</code> - Check cards for private
├──────────────────────────┤
│ <code>.gpt_img</code> - ChatGPT draw image
│ <code>.gpt</code> - Ask something in ChatGPT
│ <code>.lol</code> - Animation find-daun
│ <code>.fake</code> - Generate fake user data
│ <code>.type</code> - Animation type text
│ <code>.setname</code> - Set your name in telegram
├──────────────────────────┤
│ <code>.chatid</code> - Get a chat id
│ <code>.help</code> - Calls this menu
| <code>.spam</code> - [Spam message] [Reply to message and photo]
@Skeleton_Realm - Лучший канал в котором есть скрипты, утилиты, имногое другое!
│
╰──────────────────────────╯</b>''')




client.start()
R = '\033[31m'
G = '\033[32m'
W = '\033[0m'

banner = r'''
╔╗─╔╗╔══╗╔╗─╔╗─╔══╗╔══╗───╔══╗─╔══╗╔════╗
║╚═╝║╚╗╔╝║╚═╝║─╚╗╔╝║╔╗║───║╔╗║─║╔╗║╚═╗╔═╝
║╔╗─║─║║─║╔╗─║──║║─║╚╝║───║╚╝╚╗║║║║──║║──
║║╚╗║─║║─║║╚╗║╔╗║║─║╔╗║───║╔═╗║║║║║──║║──
║║─║║╔╝╚╗║║─║║║╚╝╚╗║║║║───║╚═╝║║╚╝║──║║──
╚╝─╚╝╚══╝╚╝─╚╝╚═══╝╚╝╚╝───╚═══╝╚══╝──╚╝──
'''
print(G + banner + W)
time.sleep(1)
print(R + "Created By :- " + G + "$¢μLL Team" +W)
time.sleep(1)
print(R + "Version :- " + G + ' 1.0' + W + '\n')

client.run_until_disconnected()
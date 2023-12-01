import viz
import vizact
import random
import vizinfo
import vizfx
import pyttsx3
import openai
import vizinput


#Enable full screen anti-aliasing (FSAA) to smooth edges
viz.setMultiSample(4)

viz.go()

#Increase the Field of View
viz.MainWindow.fov(60)


engine = pyttsx3.init()

openai.api_key = "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"

# Set the model to use
model_engine = "text-davinci-002"

info = vizinfo.InfoPanel(" Tabwanyani 'T' kuti Pre ayankhule \n Tabwanyani 'w' kut Comfo ayambe kuyenda \n Tabwanyani 'F' kut mudyese nkhunda \n Tabwanyani 'R' kuti Comfo abwerere anali \n Tabwanyani 'D' kuti Nafe yo avine \n Tabwanyani 'P' kuti Nafe apumire kuvina \nTabwanyani 'S' kut Nafe asiye kuvina \n Tabwanyani 'M' kut Tesla izizungulira",align=viz.ALIGN_RIGHT_BOTTOM,icon = False)
info.setTitle('Malamulo')
info.addSeparator()
	
ground = viz.add('sky_day.osgb')
ground.setScale(100,100,100)
ground.collidePlane()


mo = viz.addChild('hotel.osgb')
#viz.MainView.setPosition([41.8,13.33,17])

avatar = viz.addAvatar('vcc_male.cfg')
avatar.setScale(60,60,60)
avatar.setPosition([890,-25,1067])
avatar.collidePlane()
avatar.state(9)

def makepretalk():
	avatar.state(14)
	speech = vizact.speak('prenli.wav', threshold = .001, scale = 0.01, sync = True)
	avatar.addAction(speech)
	avatar.execute(14)
	#avatar.say('hello')

vizact.onkeydown('t',makepretalk)


def talktopre():
	
	user_input = str(vizinput.input('Message pre'))
	
	try:
		# Use the OpenAI API to generate a response
		completion = openai.Completion.create(
			engine=model_engine,
			prompt=user_input,
			max_tokens=1024,
			n=1,
			stop=None,
			temperature=0.5,
		)

		# Get the response from the API
		response = completion.choices[0].text
	
		engine.save_to_file(response, "prenli2.wav")
		engine.runAndWait()
	
		avatar.state(14)
		speech = vizact.speak('prenli2.wav', threshold = .001, scale = 0.01, sync = True)
		avatar.addAction(speech)
		avatar.execute(14)
		
	except Exception:
		
		engine.save_to_file(user_input, "prenli2.wav")
		engine.runAndWait()

		avatar.state(14)
		speech = vizact.speak('prenli2.wav', threshold = .001, scale = 0.01, sync = True)
		avatar.addAction(speech)
		avatar.execute(14)
	
vizact.onkeydown('q',talktopre)

avatar2 = viz.addAvatar('vcc_male2.cfg')
avatar2.setScale(60,60,60)
avatar2.setPosition([90,-25,880])
avatar2.state(5)

def walk():
	avatar2.state(2)
	walk_over = vizact.walkTo([940,-25,967],walkSpeed = 70)
	avatar2.addAction(walk_over)
	
vizact.onkeydown('w',walk)
	
def return_():
	avatar2.state(2)
	walk_over = vizact.walkTo([90,-25,880],walkSpeed = 80)
	avatar2.addAction(walk_over)
	
vizact.onkeydown('r',return_)

#avatar2.state(3)
#avatar2.execute(3)

prehouse = viz.addChild('modern.osgb')
prehouse.setScale(70,70,70)
prehouse.setPosition([-600,0,-3000])

viz.MainView.setPosition([827,96,1269])

plants = []
for z in [850,1050,1150,1250]:
	plant = viz.addChild('plant.osgb',cache = viz.CACHE_CLONE)
	plant.setScale(40,40,40)
	plant.setPosition([500,-25,z])
	plants.append(plant)

spin = vizact.spin(0,1,0,50)

#for plant in plants:
#	plant.addAction(spin)
	
def spinPlant(plant):
    plant.addAction(spin)
vizact.ontimer2(0.5,19,spinPlant,vizact.choice(plants))

pigeons = []
for p in range(13):
	
	x = random.randint(750,900)
	z = random.randint(1150,1300)
	
	pigeon = viz.addChild('pigeon.cfg',cache = viz.CACHE_CLONE)
	pigeon.setScale(40,40,40)
	pigeon.setPosition([x,-25,z])
	pigeon.state(1)
	
	pigeons.append(pigeon)
	
def pigeonsFeed():

    random_speed = vizact.method.setAnimationSpeed(0,vizact.randfloat(0.7,1.5))
    random_walk = vizact.walkTo(pos=[vizact.randfloat(750,950),-25,vizact.randfloat(1000,1350)],walkSpeed = 15)
    random_animation = vizact.method.state(vizact.choice([1,3],vizact.RANDOM))
    random_wait = vizact.waittime(vizact.randfloat(5.0,10.0))
    pigeon_idle = vizact.sequence(random_speed,random_walk,random_animation, random_wait, viz.FOREVER)

    for pigeon in pigeons:
        pigeon.addAction(pigeon_idle)

vizact.onkeydown('f',pigeonsFeed)
	
aud = viz.addAudio('hv.mp3')

avatar3 = viz.addChild('vcc_female.cfg')
avatar3.setScale(60,60,60)
avatar3.collidePlane()
avatar3.setPosition([280,-25,1150])
avatar3.state(1)
turn = vizact.turn(90)
avatar3.addAction(turn)

def dance():
	avatar3.state(5)
	aud.play()
	
vizact.onkeydown( 'd', dance )

def stop_dance():
	avatar3.state(1)
	aud.stop()
	
vizact.onkeydown( 's', stop_dance )

def pause_dance():
	avatar3.state(3)
	aud.pause()
	
vizact.onkeydown('p',pause_dance)

aud.loop( viz.ON ) 

house2 = viz.addChild('hotel.osgb')
house2.setPosition([0,0,1930])

cyber = viz.addChild('cyber.osgb')
cyber.setPosition(50,0,1300)
cyber.setScale(0.6,0.6,0.6)

def spin_():
	body = cyber.getChild('FL')
	body2 = cyber.getChild('FR')
	body3 = cyber.getChild('RL')
	body4 = cyber.getChild('RR')
	
	#body.setPosition(body.getPosition())
	#body2.setPosition(body2.getPosition())
	#body3.setPosition(body3.getPosition())
	#body4.setPosition(body4.getPosition())
	#body.addAction()
	body.addAction(vizact.spin(0,0,60,10))
	body2.addAction(vizact.spin(0,0,60,10))
	body3.addAction(vizact.spin(0,0,60,10))
	body4.addAction(vizact.spin(0,0,60,10))

def timee():
	vizact.ontimer2(0.5,19,spin_)
	
vizact.onkeydown('m',timee)

#angle = math.radians(90)

#turn2 = vizact.turn(180)
#house2.addAction(turn2)

#viz.MainView.setEuler([200,0,0])


#viz.collision(viz.ON)

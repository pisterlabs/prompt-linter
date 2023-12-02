from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.models import User
from django.shortcuts import render, redirect, get_object_or_404, reverse
from django.urls import reverse_lazy
from django.views import View
from random import shuffle
from django.core.paginator import Paginator
from django.http import HttpResponse
import openai, os
from django.shortcuts import render
from django.views.generic import CreateView, ListView, UpdateView, DetailView

from AGONy.models import Hero, Monster, Stage, Event, Origin, Comment, MonsterImage #, Game
from AGONy.forms import HeroCreateForm, MonsterCreateForm, CreateUserForm, LoginForm, OriginCreateForm, EventCreateForm

class CreateDefaultsInAgony(View):

    def get(self, request):

        """message =
            Let's just create some defaults for you to play already, who wants to go through boring creating the world?
            """

        # A view to create some defaults in game!
        Hero.objects.create(name='Percy McPerson', race=0)
        Hero.objects.create(name='Woody Oakson', race=1)
        Hero.objects.create(name='Shorty MacBeard', race=2)

        # human-like monsters, but not exactly - orcs and stuff
        Monster.objects.create(name='Zoglak the Nasty Goblin', hp=20, attack=2, defence=0, monster_level=0, monster_type=0, 
                               description="""Little, nasty green creature, filled with hate and hunger - looks like it wants to be its next meal!""")
        
        Monster.objects.create(name='Morbuk the Angry Orc', hp=40, attack=3, defence=1, monster_level=1, monster_type=0, 
                               description="""Big, angry green creature that finds you very attractive... as a food, 
                               and you guessed it - it means you should be afraid!""")        
             
        #Monster.objects.create(name='Goblin Wolf Rider', hp=50, attack=7, defence=4, monster_level=2, monster_type=0)
        #Monster.objects.create(name='Troll', hp=80, attack=8, defence=2, monster_level=3, monster_type=0)
        #Monster.objects.create(name='Giant', hp=200, attack=15, defence=5, monster_level=4, monster_type=0)
        
        #humans - but nasty ones        
        Monster.objects.create(name='Jesse James the Greedy Bandit', hp=40, attack=3, defence=1, monster_level=1, monster_type=0,
                               description="""ever heard saying: dont talk to strangers? Well, one od them just approached you, and seems like he 
                               wants to befriend you - why elese would he shout "Your money or your life!"?""")
        
        Monster.objects.create(name='Jack the Yellow-Magnetic-Star', hp=444, attack=5, defence=1, monster_level=1, monster_type=0, 
                               description="""in the beginning there was not much, but Jack was already theere. Immortal, just like alkohol. 
                               Cuts swiftly through air like a leszczyna, seems like you are his next target. Remember, you the motyka, Jack - the sun.""")


        # wild-wild-life monsters
        Monster.objects.create(name='Fluffball the Hungry Wolf', hp=25, attack=3, defence=0, monster_level=0, monster_type=1,
                               description="""Ever heard tales and stories why you 
                                should not walk into the woods? You guessed it - here comes 
                                the wolf and it counts you'll be a fancy snack!""")
        
        Monster.objects.create(name='Trinity the demon-pig', hp=25, attack=3, defence=0, monster_level=0, monster_type=1,
                               description="""You know pigs, dont you? Funny animals full of delicious bacon. Not this one. 
                               This one is full of hate and considers you its delicious bacon. Run or prepare to be eaten alive!""")
        
        Monster.objects.create(name='Spidey the Giant Venomous Spider', hp=50, attack=3, defence=0, monster_level=1,
                               monster_type=1, description="""Itsy bitsy giant venomous spider - Oh, I'm so cute:
                                I have eight long furry legs, eight terryfing eyes set on you, and you 
                                guessed it! I want some cuddels and cover you in webs and then eat! Come to papa!""")
        
        Monster.objects.create(name='Andrew the One-eyed war-experienced Octopus', hp=50, attack=3, defence=0, monster_level=1,
                               monster_type=1, description="""an octopi war veteran, tired of life, combatant and commander in 
                               Sixth Deep Sea war, now retired, and what now? You just came to spoil his rest, get to arms! 
                               Prepare for squishy hugs and chokes!""")
        
        Monster.objects.create(name='Huggy the Angry Bird-Bear', hp=80, attack=5, defence=1, monster_level=2, monster_type=1,
                               description="""Have you ever heard of angry bird? Probably. 
                               Heard of angry bear? Probably. Heard of Bird-Bear? Never? Well, some kind of 
                               psycho-druid created this abonomination, and now it's up to you to face IT. 
                               And get rid of IT. For everyone!""")

        Monster.objects.create(name='Bitsy the Courageous - Volcanic Ladybug', hp=80, attack=5, defence=1, monster_level=2, monster_type=1,
                               description="""ahh, ladybugs, such a peaceful, adorable creatures! Well, not this one - this one spits fiare 
                               and lava, and wants you to be it's next takeaway food! Yikes!""")

        Monster.objects.create(name='Ulrich the All-Seeing Beholder', hp=100, attack=5, defence=2, monster_level=3, monster_type=1, 
                               description="""I spy with my little eye, actually with my every eye, and my spying tells me you are quite 
                               a snack, come closer so I can take a little bite, you look like a tasty tasty snack! So tasty!""")
        
        Monster.objects.create(name='Killian the Fancy Rainbow-Colored Unicorn', hp=100, attack=5, defence=2, monster_level=3, monster_type=1, 
                               description="""A unicorn! What a wonderful walking real wonder! Wait, why does it chew a human body? Wait, why 
                               does it suddenly look interested in me? Wait, what? Why it runs towards me licking its tounge? Somebody please 
                               help me from this wonder! HELP! HELP!""")
        
        Monster.objects.create(name='Zygfryd the III - Proud and Wise Dragon', hp=150, attack=10, defence=3, monster_level=4, monster_type=1,
                               description="""Mystic and poud creature, but (there's always a but!) has a 
                               nasty habit - it hoards anything gold-like and shiny! (it wants a new addition 
                               to it's collection, and you guessed it - it wants you and your shines!""")

        # undead monsters
        Monster.objects.create(name='Rob the White Zombie', hp=40, attack=2, defence=0, monster_level=0, monster_type=2,
                               description="""Rumored to be a famed, well known metal star... Clumsy, stinking, brainless... those damn zombies! Ugh, one 
                               just dropped its liver. DISGUSTANG! Well, brainless for now - it wants your 
                               brainzzz! Now protect it or 
                               becomen another brainless, wiggly, rotting walking corpse!""")

        Monster.objects.create(name='Ridge Forrester the All-time Long-living Skeleton', hp=55, attack=3, defence=0, monster_level=1, monster_type=2,
                               description="""There's something there! Something white and full of calcium. 
                               Hey, why those bones hover in air? 
                               Hey, why those skull turned into my direction? Oh hell no, why it moves 
                               towards me? Shouldn't it behave nicely and just stay dead? And why it mutters something about Brooke""")
        
        Monster.objects.create(name='Marky Mark the Vegenful Spirit', hp=60, attack=6, defence=6, monster_level=2, monster_type=2,
                               description="""some spirits stay on earth even after death - mostly 
                               because their life was ended by murder or other foul action. 
                               Now you encountered one. Not a pleasant spirit this one is, oh no.""")
        
        #Monster.objects.create(name='Lich', hp=80, attack=15, defence=3, monster_level=3, monster_type=2)
        
        #Monster.objects.create(name='Vlad the Bloodthirsty Vampire', hp=125, attack=6, defence=5, monster_level=4, monster_type=2)


        #0 - escape-able monster encounter
        Event.objects.create(name='Avoidable_fight', event_type=0, event_description="""Your Hero encountered Monster, now stand and fight it! 
                             Or try to run, just to live another day! No judgement here, world is hard enough even without fighting monsters!""")
        
        #1 - loot encounter
        Event.objects.create(name='Loot_encounter', event_type=1, event_description="""Ooh, shiney! You found something! Seems like after all it was worth to walk 
                            and put yourself in all this danger. Now let's see what you found!""")
        
        #2 - trap
        Event.objects.create(name='Spikey_trap', event_type=2, event_description="""Oh crap, it's a trap! A spikey trap! A very, very spikey trap full of spikes! Of course when did you find it out? Just when you stepped
                            into that trap and sprung it! Damn, it must have hurt! How are you holding up? Still have all limbs?""")
        
        #3 - empty encounter
        Event.objects.create(name='Enjoying_views', event_type=3, event_description="""Wonderful views, aren't they? So beautiful landscape! 
                            You take a while to just enjoy this peaceful moment, after all saving the world (or conquering it, or 
                            anything else you doing, can wait a little moment)""")
        
        #2 - Trap
        Event.objects.create(name='Falling_into_cave', event_type=2, event_description="""watch your step! While watching beautiful bird you fell into a cave, a dark, dark cave. 
        Youre lucky you didnt break your legs. Anyway, escaping cave took a lot od time.""")
        
        #4 - Ambush-fight
        Event.objects.create(name='Ambush_fight', event_type=4, event_description="""While  wandering through plains you felt watched - 
        but it's too late to do anything else but fight! Draw your weapon!""")
        
        #5 - Visitable crypt - undead monsters
        Event.objects.create(name='Visitable_crypt', event_type=5, event_description="""wandering through Forest you notice more and more dead trees. Then, you 
        notice why - you stumble upon an old and grim crypt - do you dare to enter IT?""")        
       
        #6 - Healing encounter
        Event.objects.create(name='Rest_at_oasis', event_type=6, event_description="""Amidst nowhere, you found a beautiful, blossoming oasis. You 
        take a well-deserved sip of crystal-clear water and at instant feel refreshed and more vigorous.""")

        return redirect('AGONy_index')  #, {'message': message})

    

        # origins - general
        #Origin.objects.create(origin_type=0, origin_description="Tired of mundane life, felt call for adventure")
        #Origin.objects.create(origin_type=0, origin_description="Want to get rich fast, or die trying")
        #Origin.objects.create(origin_type=0, origin_description="Got lured to adventure by songs of riches and glory")
        #Origin.objects.create(origin_type=0, origin_description="Broke the law, it's desperate try to clear name")

        # origins - tragic 
        """
        Origin.objects.create(origin_type=0,
                              origin_description="Got his family murdered, now on a quest to avenge them!")
        
        Origin.objects.create(origin_type=0,
                              origin_description="Murdered a lot of people, now running away from law enforcement!")
        
        Origin.objects.create(origin_type=0, origin_description="Got set up in criminal activity, it's a way to repent")
        """
    
"""['\n\nHonestly, I knew that I would have to do some work someday to earn my fame, but I never
 imagined how hard it would be', ' Today I met my first Monster', ' But not just any monster: 
 it was an evil pig', " I don't know how it got here but I had the perfect plan, I would outsmart it by
  using my cat, the purrfect weapon to take out a pig", ' I reached into my inventory, whipped out my 
  cat, and it knocked the pig into the nearest tree', ' Success!\n\nThe pig just barely got 
  up and looked at me', ' "Must be nice to have a cat for a weapon, eh? You\'ll see who\'s better 
  in a fight with a mustache like mine', '"\n\nThe pig charged at me and I opened the inventory 
  again and got out the boomerang to attack', ' The boomerang hit the pig in the face and 
  knocked it over', ' I was so excited that I screamed and ran up to grab'] """
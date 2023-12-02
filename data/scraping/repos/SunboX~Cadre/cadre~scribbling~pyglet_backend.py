# Licensed under the MIT license
# http://opensource.org/licenses/mit-license.php

# Copyright 2009 Frank Scholz <coherence@beebits.net>

import os

# Twisted
from twisted.internet import reactor

from coherence import log
# Pyglet

import pyglet


class Actor(object):

    def __init__(self, drawable=None):
        self.drawable = drawable

    def replace(self, drawable):
        self.drawable = drawable


class Canvas(log.Loggable):

    logCategory = 'canvas'

    def __init__(self, fullscreen=True):
        self.fullscreen = fullscreen
        self.transition = 'FADE'

        self.parts = []
        self.stage = pyglet.window.Window()
        if self.fullscreen == True:
            self.stage.set_fullscreen(True)
        else:
            self.stage.set_size(1200, 800)

        self.width, self.height = self.stage.get_size()

        self.display_width = int(self.width * 0.7)
        self.display_height = int(self.height * 0.7)

        self.display_pos_x = float((self.width - self.display_width) / 2)

        self.in_texture = Actor(None)
        self.out_texture = Actor(None)

        self.parts.append(self.in_texture)
        self.parts.append(self.out_texture)

        @self.stage.event
        def on_draw():
            self.stage.clear()
            for part in self.parts:
                if part.drawable != None:
                    part.drawable.draw()

    def set_title(self, title):
        self.stage.set_caption(title)

    def process_key(self, stage, event):
        print "process_key", stage, event

    def get_available_transitions(self):
        return [str(x.replace('_transition_', '')) for x in dir(self) if x.startswith('_transition_')]

    def _transition_NONE(self):
        self.out_texture.replace(None)
        self.out_texture, self.in_texture = self.in_texture, self.out_texture

    def load_the_new_one(self, image, title):
        self.warning("show image %r", title)
        if image.startswith("file://"):
            filename = image[7:]
        else:
            #FIXME - we have the image as data already, there has to be
            #        a better way to get it into the texture
            from tempfile import mkstemp
            fp, filename = mkstemp()
            os.write(fp, image)
            os.close(fp)
            remove_file_after_loading = True
        self.warning("loading image from file %r", filename)
        pic = pyglet.image.load(filename)

        new_in_texture = pyglet.sprite.Sprite(pic, self.display_pos_x, self.height - self.display_height - 20)
        #print "sprite",new_in_texture.width,new_in_texture.height,new_in_texture.scale
        pic_width, pic_height = new_in_texture.width, new_in_texture.height
        new_in_texture.scale = float(self.display_height) / float(pic.height)
        #print "after h_scale",new_in_texture.width,new_in_texture.height,new_in_texture.scale
        if new_in_texture.width > self.display_width:
            new_in_texture.scale = float(self.display_width) / float(pic.width)
            #print "after w_scale",new_in_texture.width,new_in_texture.height,new_in_texture.scale
        new_in_texture.set_position((self.width - new_in_texture.width) / 2, self.height - new_in_texture.height - 20)
        self.in_texture.replace(new_in_texture)
        self.set_title(title)
        try:
            if remove_file_after_loading:
                os.unlink(filename)
        except:
            pass

    def show_image(self, image, title=''):
        self.load_the_new_one(image, title)
        function = getattr(self, "_transition_%s" % self.transition, None)
        if function:
            function()
            return
        self._transition_NONE()

    def add_overlay(self, overlay):
        pass
